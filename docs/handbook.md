# Parameter Decomposition — Handbook

The single source of knowledge on parameter decomposition — interpretability in parameter space: why mechanisms are sought in the weights rather than the activations, what a parameter decomposition is and the properties that make one trustworthy, the evidence standards and failure modes, and what a finished decomposition licenses downstream (subcomponent interpretation, attention analysis, attribution graphs, clustering, model editing). The [parameter-decomposition skill](skill.md) carries the executable recipe; this handbook carries the science. Parameter decomposition is causal ablation carried out in *parameter* space.

## Summary

Parameter decomposition splits a neural network's weights into parameter components — vectors in parameter space — such that each component implements a small piece of the network's learned algorithm, only a few components are needed on any given input, and the unneeded components can be ablated, in any combination, without changing the output. It lets us say *"these are the parts of the weights that this behavior causally depends on"*.

It differs from activation-space interpretability: instead of using featurizers first to look for the representations (variables) on which computations might be done, parameter decomposition identifies the objects doing the computations first, and then asks what representations they use. It uses a learned *causal importance function* to estimate the causal importance of each parameter component on a given input. The causal importance is the degree to which a parameter component *cannot* be ablated without affecting the output (1 = not ablatable at all; 0 = fully ablatable).

## Why decompose parameters

**Architectural units are not mechanisms.** Neurons, attention heads, and layers do not reliably map to individual, interpretable computations: representations span neurons, spread across heads, and stretch over layers. A neuron-by-neuron account of a network is perfectly accurate and useless — unnecessarily long, phrased in polysemantic parts, carving the network nowhere near its joints.

**Activation-based decompositions train replacement models.** Sparse dictionary methods (SAEs, transcoders, cross-layer transcoders) fit a *different* function — wider, differently non-linear — to the transitions between activations. This has produced real insight into intermediate representations, but the account it gives is hard to relate to the objects doing the computing: the network's parameters and nonlinearities. The mismatch is practical, not just aesthetic: a replacement model with more representational capacity than the original can learn representations that the original could not computationally distinguish (feature splitting is one symptom), and it gives no direct handle for making precise, predictable *edits* to the model's algorithm.

**Parameter space is mechanism space.** During learning, gradient descent iteratively etches a neural network’s mechanisms into its parameter vector, which makes it natural to look for mechanisms in parameter space. Parameter space satisfies several necessary properties that 'mechanism space' must have: it spans the functional range from "does everything the network does" (the full parameter vector) to "does nothing" (the zero vector); it accommodates mechanisms that are not aligned with neurons, heads, or layers; it accommodates computation in superposition (more mechanisms than neurons); and it accommodates multidimensional mechanisms (mechanisms that operate on multidimensional representations, such as multidimensional manifolds).

**What the stance buys.** It strives to identify mechanisms, i.e. *steps in the network's algorithm*. A non-circular footing for "features": features are *properties of the input that activate particular mechanisms* — a definition that does not depend on human interpretability or choice of featurizer. Structural immunity to feature splitting: subcomponents marked causally unimportant must be ablatable *in any combination*, so the decomposition cannot invent ever-finer sparsely-activating parts the way a dictionary can — a split part's siblings would be jointly ablatable only when none of them is real (and this is verified empirically: added capacity goes unused rather than splitting). Architecture-agnosticism: attention layers decompose by the same procedure as MLPs, and the resulting parts routinely span multiple attention heads — distributed attention computations become analyzable instead of invisible. And a direct substrate for editing: the parts are weights, and weights are the network, which is what we wanted to understand, rather than the dataset on which it operates.

## What a parameter decomposition is

### The objects

Each weight matrix \(W^l\) is decomposed into a sum of **rank-one subcomponents** \(\vec{U}^l_c (\vec{V}^l_c)^\top\) — each a "read" direction \(\vec{V}\) paired with a "write" direction \(\vec{U}\). The number of subcomponents per matrix may exceed the matrix's rank; that headroom is exactly what lets the decomposition capture mechanisms in superposition. **Components** are clusters of subcomponents that tend to be causally important together; a component may span several matrices and layers, and is the object that deserves the name "mechanism". Subcomponents are not components — but individual subcomponents are often interpretable on their own, and much analysis proceeds at the subcomponent level. **Δ-components** parametrize the residual between the summed subcomponents and each target (original) matrix; they are defined to be causally unimportant everywhere and tend to be small.

Bias parameters in the target model are currently left undecomposed. Their causal importance is fixed at 1 on every input.

A **causal importance function** — itself a trained neural network — maps the target model's hidden activations to a **causal importance value** \(g^l_c \in [0,1]\) for every subcomponent at every sequence position. Causal importance is defined by *ablatability*: \(g = 0\) means the subcomponent can be fully or partially ablated on this input without changing the model's output; \(g = 1\) means it cannot be ablated at all. Ablations are implemented with masks \(m = g + (1-g)r\) with \(r \in [0,1]\), so an important subcomponent's mask is pinned to 1 while an unimportant one may be scaled anywhere down to 0.

### The four properties

A decomposition is trained toward four properties. If it exhibits all four, its components are strong candidates for the network's mechanisms.

- **Parameter faithfulness** — the components sum to the target network's parameter vector. A *static* property of the decomposition, checked by direct comparison (and by the unmasked model matching the target). This is stricter than behavioral equivalence: different parameters can produce the same behavior, but only the actual parameters are the actual mechanisms.
- **Minimality** — as few subcomponents as possible are causally important on any given input. This is what makes the decomposition a *simplification*: each datapoint's computation is described by a small subset of components.
- **Mechanistic faithfulness** — every subset of components that includes the causally important ones computes the network's output: the causally *un*important components are ablatable **in any combination**, partially or fully, *including combinations chosen adversarially to break the output*. A *behavioral* property, checked by intervention. Keep it distinct from parameter faithfulness — one says the components add up to the weights, the other says the importance labels survive being acted on.
- **Simplicity** — each component contains as little computational machinery as possible. The rank-one constraint carries most of this, plus a slightly superlinear penalty on how **frequently** a subcomponent is important — without it, unrelated mechanisms used on disjoint data can hide inside a single rank-one subcomponent (their sum can itself be rank one). In the current JAX objective this frequency pressure has its own coefficient and reference-token count; it is not a `beta` nested inside the importance penalty. Description length of the *parameters* used per datapoint is the operative measure, and it is just a proxy for what we ultimately want — to minimize the description length of the *computation*.

**Why "ablatable in any combination" is the defining requirement.** Two failure cases show why weaker criteria don't suffice.

- *Joint ablation only.* Suppose unimportant components only had to be ablatable *all together*. Then a pair of components whose contributions cancel could both be labeled unimportant — ablating both together changes nothing — even though each is individually load-bearing: ablate either one alone and the output changes. The labels are wrong, and anything downstream that trusts them (an edit that removes one "unimportant" component, a circuit account that drops it) breaks the model. Demanding output invariance along *every* partial-ablation path catches the pair and correctly forces both labels to "important".
- *No ablation-combination requirement at all.* Drop the requirement and a spurious decomposition scores perfectly on every other property: invent one low-rank component per datapoint, tuned to reproduce that datapoint's output, plus one residual component to restore parameter faithfulness. This giant lookup table of the training set is perfectly parameter-faithful, *maximally* minimal (exactly one component important per input — minimality actively rewards it), and even survives the joint-ablation check, since each datapoint's component reproduces the output on its own. Only any-combination ablatability kills it: a permitted combination that leaves other datapoints' shards partially unablated sums to an arbitrary parameter vector and wrecks the output. Components must not interfere with the computation on data they are not important for — so general machinery cannot be split into memorized shards.

The same property is what allows explanations to **aggregate**: if two inputs are each explained by their own subset of components, the union of the subsets still (approximately) computes both — so local explanations of single behaviors can compose into global accounts of the model.

**Why adversarial checking is part of the definition.** All possible mask combinations cannot be enumerated, so the property is checked by sampling. Stochastic (uniform) sampling bounds the *average* case; the property is about *every* case, and decompositions trained only against stochastic masks reliably harbor mask settings — findable by a gradient-based adversary — that destroy the output while touching only "unimportant" components. Adversarially-chosen ablations are therefore part of what mechanistic faithfulness *means* operationally, not merely an optional hardening.

### Two senses of "active"

Keep two notions apart when reading any analysis. **Causal importance** (\(g\)) says the subcomponent is *needed* — but it is not a local measure; a subcomponent can interact strongly with the activations and have its effect suppressed downstream. **Subcomponent activation** (\(a^l_c = \lVert\vec{U}^l_c\rVert (\vec{V}^l_c)^\top \vec{\varphi}^l\)) measures the local interaction between the input activations and the subcomponent's read direction, whether or not it ends up mattering. Superposition guarantees more interactions than important interactions, so high activation with zero importance is common and meaningful (interference), and interpretation work should lead with causal importance.

*Terminology note:* earlier papers in this line use "attribution" for a gradient-based estimate of what is now measured directly as causal importance, and use "parameter component" for a full-parameter-vector object rather than today's cluster-of-subcomponents. This handbook and the skill use the current terms throughout.

## The parameter-decomposition pipeline

**Input.** A target model; the set of weight matrices to decompose; the data distribution the decomposition should hold over (usually the model's training distribution — the mechanisms found are relative to it); and the claim or downstream use the decomposition serves.

**Pipeline.**

1. **Scope the claim and calibrate the instrument.** What behavior or model is being decomposed, and what will the decomposition be used for — a mechanism inventory, a circuit question, an edit? The downstream use fixes the data distribution (an external dataset for LMs; a synthetic generator for toys) and the matrices worth decomposing. In a new domain, you should first decompose a target whose mechanisms are already known and verify that the method recovers them. This anchor is mandatory: under-training and an ill-matched reconstruction objective can both produce clean-looking nulls, so an unanchored target cannot distinguish between 'real absence of structure' and 'badly configured method'.
2. **Decompose, to convergence.** Train the subcomponents and causal importance function, sweeping the sparsity pressure (importance-minimality coefficient) — the right value is not knowable a priori. Runs must reach convergence; the sparsity penalty's schedule (Smooth-L0 `gamma`, or `pnorm` for the Lp variant) means minimality arrives late, and unconverged decompositions are not comparable to each other or to anything else.
3. **Validate against the evidence standards** (next section). A decomposition that fails them is not a weaker result to be reported with caveats; it is not yet a decomposition.
4. **Cluster subcomponents into components** where the analysis needs mechanism-level objects; skip when subcomponent-level analysis suffices (it often does).
5. **Analyze** (see *Working with a decomposition*): interpret subcomponents, trace circuits with attribution graphs, analyze attention behaviors, edit.
6. **Report** the tradeoff surface and the checks, not one flattering checkpoint.

This repository carries the executable workflow for steps 2–3 and the later-stage tools. The [parameter-decomposition skill](skill.md) carries sweep design, smoke criteria, convergence checks, and result selection.

## Evidence standards

A good decomposition shows *worst-case* reconstruction under the strongest evaluation the domain implements **and** genuine minimality — the number of causally important subcomponents per datapoint (\(L_0\)) is small relative to a domain-appropriate baseline. Matrix rank is a useful scale for some LM sites, not a universal threshold: for a toy with known mechanisms, compare against the ground-truth number used by an input (for TMS 5→2, typically about one active feature), not against rank 2. Both must hold: low reconstruction error with needlessly high \(L_0\) is faithful but no simplification; low \(L_0\) with high reconstruction error is sparse but unfaithful. Verify parameter faithfulness separately: the *unmasked* model (all masks 1, Δ-components excluded) must match the target — if it doesn't, the components don't even sum to the model being explained.

**The evaluation adversary is deliberately shaped.** Fresh adversarial masks at evaluation are stricter than the persistent adversary used in training, and they are *shared across the batch* on purpose: a fully per-datapoint adversary can exploit uncorrelated superposition-interference noise, so sharing forces it to find *systematic* defects in the decomposition rather than noise-fit single points.

**Monitor, never optimize.** Reconstruction under causal-importance-values-as-masks, and under rounded (binarized) importances, indicate whether the components labeled important suffice on their own. They are informative *only because nothing optimizes them*: both are trivially driven to perfection if included in training, without the decomposition capturing anything. The adversarial loss (and, less so, the stochastic loss) is the term that resists this cheating.

**How much adversarial robustness to demand is open.** Complete robustness is provably too strict: an unconstrained adversary can co-ordinate the interference noise of genuinely-inactive superposed circuits (ablate all negative-noise contributors, keep all positive ones) and change the output of a decomposition we should accept. The practical standard: the decomposition should be robust to the kinds of ablations you will actually perform — the component maskings used in analysis and editing, over the data you care about — rather than to arbitrarily strong per-datapoint attacks. Robustness to a modest adversarial budget, degrading gracefully as the budget grows, is the expected signature of an acceptable decomposition; report the budget alongside the number.

**Convergence and metric discipline.** The minimality proxy sharpens over training (`gamma` decreases for Smooth-L0; `p` decreases for the Lp variant), so \(L_0\) falls steadily late into the run and the adversary takes many steps to bite — most of a decomposition's apparent quality is unreadable before convergence. End-of-run metric values are noisy point estimates; compare runs on windows of settled values, never single points, and treat the adversarial metric as the noisiest of all (read its plateau, not its last value).

**Reconstruction is scored in the target's own output metric** — KL per position for a language model, MSE for TMS or ResidMLP, whatever a new domain's output space calls for. Report which metric a reconstruction number is in; values are not comparable across metrics, and a new domain earns its recon metric before it earns a Pareto front.

**Minimality needs an anchor.** Use the strongest baseline the domain permits:

| Domain | Minimality anchor |
| --- | --- |
| Closed or exhaustively verifiable target | Exact task accuracy under masking over the full input space, compared with chance; report the smallest sufficient set only among masks that preserve the target's exact accuracy. |
| TMS or another target with known mechanisms | Ground-truth number and identity of mechanisms used by each input. |
| Language model without mechanism ground truth | The reconstruction/minimality Pareto front, with matrix rank and the dense all-subcomponents count as scales rather than universal thresholds. |
| Classifier without mechanism ground truth | Architectural units (neurons, channels, or heads) under matched causal masking. Also target model- and chance performance. |

A fair comparison with neurons or heads holds the target, data, mask family, reconstruction metric, and adversarial budget fixed; it compares active-unit count at matched reconstruction (or reconstruction at matched count), reports each method's unit granularity, and includes the dense and chance endpoints. Merely beating the architectural-unit count is not evidence of useful minimality: even an untrained hundred-step smoke can often clear a threshold that weak. If a bad or untrained baseline passes the stated bar, strengthen the bar before interpreting the decomposition.

**Comparing decompositions and methods.** Compare on the tradeoff between sparsity and (eval) adversarial reconstruction — Pareto fronts across the sweep — and be explicit about the sparsity notion, since "one active unit" means different things across methods (a per-matrix subcomponent, a per-layer latent, a cross-layer latent). Interpretability comparisons use held-out representative examples and causal interventions; semantic labels remain hypotheses until those checks support them. Thresholding tiny importances (e.g. 0.1) is legitimate — they carry almost no reconstruction. Feature-splitting tests: grow capacity and watch whether the number of alive subcomponents grows with it (splitting) or stays flat (headroom unused — the healthy signature).

Seed-consistency (mean max cosine similarity across decomposition seeds) is evidence but not the goal: networks are highly degenerate, and there may be a *space* of mechanistically faithful decompositions rather than one ground truth — faithfulness, not uniqueness, is the property to insist on. Measure this first across decomposition seeds of the **same frozen target**. Before comparing decompositions of independently trained targets, characterize the target-training variance itself (task metrics, learned function, and transition timing where applicable); otherwise target non-reproducibility is indistinguishable from decomposition instability.

**Quantify and report.** Cite every number to the run and artifact that produced it; report the sweep's tradeoff curve, the convergence evidence, the adversarial budget, and the monitor-only metrics — not a single selected checkpoint. Plot their curves over training — don't only report point estimates at the end.

## Failure modes and false conclusions

Diagnose these before interpreting anything; several masquerade as findings about the model.

- **Causal-importance collapse.** Sparsity pressure too high: importance values sink below 1 everywhere, reconstruction blows up. Observable keys: `CIHistograms` has no mass near 1, `CIMeanPerComponent` has no alive shoulder, and eval `PGDReconLoss` rises. No subcomponent sitting at \(g \approx 1\) for most of its firing inputs is a tell.
- **Too-dense decomposition.** Sparsity pressure too low: `CI_L0` remains near the dense baseline at settled eval `PGDReconLoss`, with no useful simplification on the sweep's reconstruction/minimality Pareto front.
- **Polysemantic subcomponents.** Insufficient frequency pressure lets unrelated mechanisms share one rank-one subcomponent (often visible in per-subcomponent exemplars as two unrelated jobs keyed to the sign of its activation, while `ComponentActivationDensity` shows the component alive across both input classes). Raise the separate frequency-minimality coefficient to split them; it trades off against the importance pressure. Do not emulate this with the removed Torch-era `beta`.
- **Under-training read as "no structure".** An unconverged decomposition looks like a null result — `CI_L0` or eval `PGDReconLoss` are still falling, and the scheduled minimality shape parameter (`gamma` or `pnorm`) has not spent a settled window at its final value. Most apparent parameter-decomposition nulls are under-training; confirm these curves plateau before drawing any conclusions.
- **Privileged input basis mistaken for learned structure.** If the input representation is sparse in some basis, every readout looks structured in that basis; the test is per-unit selectivity against a random-direction control — a shuffled null does not catch it.
- **Dead subcomponents mistaken for a mechanism count.** The subcomponent budget \(C\) is deliberately over-provisioned; `CIMeanPerComponent` should show the alive/dead cutoff used for the estimate. The raw budget, `ComponentActivationDensity`'s count of units that ever fire, and a single thresholded checkpoint all mislead.

If the sweep fails to identify a good decomposition, find better hyperparameters that work. They almost certainly exist.

## Working with a decomposition

What analysis a validated decomposition permits. This repository documents the runnable tooling; the doctrine is here.

**Interpreting subcomponents.** Subcomponents activate for coherent categories of input, and are labeled from representative examples, with causal importance in place of activation magnitude and low-importance firings treated as background (they are mostly interference). Labels are hypotheses to test with held-out examples and causal interventions. Read both senses of "active" side by side: high activation with low importance is itself informative (the model *touches* the subcomponent but suppresses its effect).

**Attention analysis.** Attention parameters decompose like everything else, and the resulting subcomponents routinely span multiple heads — making computations distributed across heads analyzable. The QK circuit factors into *pairs* of query- and key-subcomponents: a static, data-independent interaction strength (how strongly a pair contributes to attention scores, as a function of position offset) surfaces behaviors like previous-token and syntax-boundary attention; a data-dependent interaction strength decomposes any actual attention pattern into per-pair contributions that can be included, excluded, and ablated one at a time. OV circuits are compared across heads by the overlap of the subspaces they read from and write to, weighted by where the data actually varies.

**Attribution graphs.** Circuits are traced by attributing each subcomponent's activation to upstream subcomponents — gradient × source activation × source importance, with gradients *stopped* through every non-source subcomponent so edges measure direct effects only. Two disciplines: (1) pruning a graph to a specific behavior must itself use adversarial sampling — graphs pruned with importances-as-masks alone are much smaller, reconstruct the behavior well, and are mechanistically unfaithful (adversarial masking of their discarded remainder destroys the prediction); (2) an attribution graph is *not* a computational graph — it records how strongly information flowed, not the function computed, and the nonlinear interactions between subcomponents at neurons remain the open frontier between current practice and full reverse engineering.

**Clustering subcomponents into components.** Where mechanism-level objects are needed (and they so far have only rarely been needed), cluster subcomponents by co-importance: binarize causal importances, treat a group's importance as the OR of its members', and minimize a description-length cost that trades the index cost of a larger dictionary against the rank cost of fatter components, with a resolution knob (α) setting where that trade sits. Merging proceeds hierarchically and stochastically to escape greedy local minima. Clustering is a separate, potentially substantial compute cost at large subcomponent budgets; budget it independently from decomposition training and record the sampling/restart budget. It is post-hoc and its resolution is a choice — do not report a component count as a fact about the model without reporting the knob.

**Model editing.** Each subcomponent's rank-one structure gives one read direction and one write direction, and either can be surgically replaced — e.g. pointing a write vector at a chosen unembedding direction rewrites what the mechanism *does* when it fires. Validate edits like interventions: measure the target behavior change *and* off-target divergence from the unedited model both near the edit's firing sites and globally. Hand edits currently trade off against trained baselines (a well-fed LoRA can win on off-target damage); the edit's value is interpretability and label-free precision, not dominance — and hybrid schemes are unexplored.

**Training diffs.** Subcomponents are directions in parameter space, so any gradient update or finetuning diff can be projected onto the decomposition's basis — expressing what training did as upweighting, modifying, or creating mechanisms. Doctrine-only for now: no shipping recipe.

## Limits of interpretation

Allowed:

- "These subcomponents are causally important for this behavior on this distribution, under maskings up to the stated adversarial budget."
- "This subcomponent is causally important on inputs of type X and ablatable elsewhere."
- "The causally important set is minimal and sufficient in the tested combinations; the unmasked decomposition reproduces the target model."
- "This edit changed the target behavior with off-target divergence of Y (surrounding) and Z (global)."
- "This attention behavior is implemented by these subcomponent pairs, distributed across these heads."

Avoid:

- "This component *is* the mechanism for X" from a decomposition alone — component identity depends on the clustering resolution, and semantic labels are hypotheses until tested like any feature label.
- "The model has N mechanisms" — dead-subcomponent counts depend on the budget, component counts on the clustering knob, and degeneracy may permit many valid decompositions.
- "The model has no parameter-space structure here" from an unconverged run or a single sweep point.
- Reading an attribution graph as the algorithm — flow strength is not functional form.
- Treating an importances-as-masks subgraph, here or in any masking-based method, as a faithful circuit without adversarial validation.
- "The decomposition is the unique ground truth" from seed consistency — faithfulness is the standard; uniqueness is not established.

## Scope and maturity

VPD's tuning maturity and validated results currently extend through 67M transformer language models, plus TMS and ResidMLP toys. Past that scale, do not summarize failure as a single size ceiling: say whether the observed blocker is **optimization** (no converged reconstruction/minimality Pareto point), **memory or runtime** (the intended run cannot complete), or **transfer** (settings from a smaller target converge but no longer yield a useful decomposition).

The library's architecture coverage is broader than that evidence base: it ships runnable Llama- and Qwen-family 8B target implementations and decomposition recipes. Those 8B seats show that the architecture and execution path exist; they do not establish tuned, trustworthy 8B decompositions. Other architectures and datasets may require significant implementation and tuning work. But using parameter decomposition at scale is an open research frontier.

## Background

- *(Key paper) Interpreting Language Model Parameters* (Bushnaq et al., 2026) — the current method (VPD) and the downstream analyses summarized here: [https://static.goodfire.ai/vpd-blog-post/post.md](https://static.goodfire.ai/vpd-blog-post/post.md)

Cite it as:

```bibtex
@misc{bushnaq2026interpreting,
  title={Interpreting Language Model Parameters},
  author={Bushnaq, Lucius and Braun, Dan and Clive-Griffin, Oliver and Bussmann, Bart and Hu, Nathan and Ivanitskiy, Michael and Linsefors, Linda and Sharkey, Lee},
  journal={Technical Report},
  institution={Goodfire and MATS},
  month={April},
  year={2026},
}
```

- *Stochastic Parameter Decomposition* (Bushnaq, Braun & Sharkey, 2025) — rank-one subcomponents and the causal importance function. arXiv:2506.20790
- *Interpretability in Parameter Space: Minimizing Mechanistic Description Length with Attribution-based Parameter Decomposition* (Braun et al., 2025) — the minimum-description-length framing and the faithfulness/minimality/simplicity criteria. arXiv:2501.14926
