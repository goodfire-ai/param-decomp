# FSDP Training Process And Scaling Strategy

This note is based on the live implementation paths, not the existing FSDP
planning/report documents.

## Target Goal

The goal is not to find a "natural root" for its own sake. The goal is to make
this repo able to run decompositions on models up to roughly 10B parameters
across many GPUs/nodes, while keeping the training code clear enough that new
losses, mask strategies, CI functions, and analysis paths can still be modified
without rediscovering FSDP constraints each time.

For that target, the important properties are:

- large target weights, component weights, gradients, and optimizer states are
  sharded
- full parameter materialization has a short and predictable lifetime
- communication can overlap with useful computation
- dense replicated intermediates such as full `W_target - V @ U` deltas are not
  created during training
- activation memory is controlled separately with checkpointing/rematerialization
- checkpoint save/load works without requiring every rank to materialize
  everything on GPU
- the distributed training path has a small number of explicit entry points
  instead of arbitrary reads from parameter-bearing submodules

An FSDP root is useful only if it helps those properties.

## Current Training Step

The main path is `param_decomp/run_param_decomp.py::optimize`.

Setup:

1. The target model is frozen with `target_model.requires_grad_(False)`.
2. `ComponentModel` replaces configured target submodules with fused
   `DecomposedLinear` / `DecomposedEmbedding` sites.
3. Each site owns trainable `V` and `U`, plus the frozen original target
   linear/embedding module.
4. If `parallel_strategy == "fsdp"`, `fsdp_wrap()` applies FSDP2 with
   `fully_shard()` in place.
5. The optimizer is created after FSDP wrapping, over trainable component
   params plus CI-function params.

The current FSDP wrapping shards:

- every `DecomposedLinear` and `DecomposedEmbedding`
- every CI-function `TransformerBlock`
- the `GlobalSharedTransformerCiFn` module itself

It does not shard:

- `ComponentModel`
- `target_model`
- target transformer block modules
- target params outside decomposed sites, such as `wte`, `lm_head`, final norm,
  and any target module not matched by `module_info`
- layerwise/non-transformer CI functions

Each training iteration then does:

1. Zero optimizer grads and update LR.
2. If a non-FSDP run includes `FaithfulnessLossConfig`, compute
   `component_model.calc_weight_deltas()` for that loss only. Under FSDP this
   metric/warmup path is now rejected because it still needs full dense deltas.
3. Run `wrapped_model(batch, cache_type="input")`.
   With no `mask_infos`, every decomposed site delegates to its frozen target
   module. This produces the target output and caches pre-weight activations.
4. Run `component_model.calc_causal_importances(...)`.
   For global transformer CI, this calls the FSDP-wrapped CI function directly.
5. Run persistent-PGD warmups if configured. Each warmup calls
   `model(batch, mask_infos=...)`.
6. Run every configured loss. Reconstruction-style losses usually call
   `model(batch, mask_infos=...)` one or more additional times. If
   `use_delta_component=True`, these losses pass only a per-site `delta_mask`;
   the decomposed site computes `target_out - full_component_out` locally while
   its own target and component params are gathered.
7. Compute PPGD source grads with `torch.autograd.grad(..., retain_graph=True)`.
8. Backpropagate `total_loss`.
9. Step PPGD sources, clip grads, and call `optimizer.step()`.

So one optimizer step can include several full target-model forwards before a
single backward. FSDP communication is paid on each of those forwards.

## What An FSDP Root Means

FSDP2 wraps individual modules. At runtime it identifies an outermost FSDP
module for a forward. That outermost module is the FSDP root for that forward.
FSDP modules reached under it in the module tree are child FSDP units.

For example:

```text
FSDP(target_model)                 root
  FSDP(h.0.attn.q_proj site)       child
  FSDP(h.0.attn.k_proj site)       child
  FSDP(h.0.mlp.c_fc site)          child
```

With FSDP2 defaults, non-root child units reshard after forward. The root unit
does not, because PyTorch assumes the root's full params may be needed
immediately in backward.

The current target path is closer to:

```text
ComponentModel                     not FSDP
  target_model                     not FSDP
    FSDP(h.0.attn.q_proj site)     independent root
    FSDP(h.0.attn.k_proj site)     independent root
    FSDP(h.0.mlp.c_fc site)        independent root
```

Since there is no FSDP-wrapped ancestor above these sites, each site can become
its own FSDP root when execution reaches it. With default `reshard_after_forward`
behavior, full site params can remain materialized after that site's forward
until backward instead of being freed immediately.

This does not mean FSDP has no benefit. Sharded params are still sharded before
forward and after backward, gradients are reduce-scattered, and optimizer state
is sharded because the optimizer is built after wrapping. The lost benefit is
the ideal "only the current layer's full params are resident" peak-memory shape.

## What A Root Actually Buys

Having a root is not automatically more scalable. Explicit
`reshard_after_forward=True` on every large FSDP unit can recover much of the
peak-memory benefit even without changing the root structure. A root helps when
it gives FSDP and the training code a coherent boundary for a whole parameter
using computation.

Concrete benefits:

- **Child-unit resharding by default.** Large nested units can gather, run, and
  reshard after their forward instead of behaving like independent roots that
  keep full params live until backward.
- **Shared ordering for prefetch.** FSDP can track a forward order across child
  units and use that order for backward prefetch. Explicit prefetch policies are
  also easier to reason about when the units are under one root.
- **Fewer accidental sharded-param reads.** If the distributed training API is
  "call this rooted forward request", code is less likely to read `V`, `U`,
  target weights, or CI params while they are sharded.
- **Cleaner checkpoint/state handling.** A coherent module tree gives state-dict
  and optimizer-state code a single place to start. It does not make
  checkpointing free, but it reduces special cases.
- **A place to attach 2D/HSDP policy.** For many nodes, we may want to shard
  within a node or node group and replicate across groups. That is easier if the
  decomposition step has a clear distributed boundary.

Limits:

- A root does not shard activations.
- A root does not remove the cost of repeated forwards in one optimizer step.
- A root does not help if large tensors are materialized manually outside
  forward. The delta-component reconstruction path no longer does this, but
  faithfulness still would without a site-local rewrite.
- A non-empty root can make memory worse if it owns large params and keeps them
  unsharded after forward.

So the root question should be evaluated by peak memory and communication
behavior, not aesthetics. The highest-impact scalability wins are still:
site-local delta math, explicit resharding of large units, activation
checkpointing, and avoiding many redundant full-model forwards.

## Should `ComponentModel` Be The Root?

It is possible in principle, but a full-param `ComponentModel` root is not the
cleanest first choice.

Advantages:

- It would give FSDP one root above all target sites and the CI function.
- Nested target sites would become child FSDP units and reshard after forward by
  default.
- FSDP would have one shared ordering context for prefetch/backward bookkeeping.

Problems:

- `ComponentModel.forward()` only runs the target model. It does not include the
  subsequent direct call to `component_model.calc_causal_importances()`. So it
  is not a natural root for the whole training step.
- If the root owns params that are used outside `ComponentModel.forward()`, those
  params can be accessed while sharded. This is a real risk for CI variants that
  are not separately FSDP-wrapped.
- If the root owns leftover frozen target params, it may force FSDP ownership of
  embeddings, final norm, and `lm_head`. That can introduce DTensor/Tensor
  boundary issues in target internals and adds all-gather work for frozen params.
- A non-empty root group defaults to keeping its full params after forward unless
  `reshard_after_forward=True` is explicit.

A more plausible use of `ComponentModel` as root would be an empty/coordinator
root: wrap it while ignoring all params not already assigned to child FSDP
groups. That could give children a shared FSDP parent without making leftover
target or CI params FSDP-owned. It needs a focused smoke test because this code
uses direct submodule calls outside `ComponentModel.forward()`.

## Making `ComponentModel` A Real Root

If breaking changes are acceptable, `ComponentModel` can become the long-term
FSDP root. To make that coherent, it has to stop being a lightweight wrapper
with public parameter-bearing internals and become the differentiable execution
engine for the whole decomposition step.

The invariant should be:

> During distributed training, no code outside `ComponentModel.forward()` calls a
> parameter-bearing method or reads parameter values from `target_model`,
> `components`, or `ci_fn`.

This does not mean the optimizer cannot hold parameters. It means all tensor
computations that need full parameter values happen under FSDP-managed forwards.

Today the invariant is false in several ways:

- non-FSDP faithfulness still calls `component_model.calc_weight_deltas()`;
  FSDP rejects that path for now.
- `run_param_decomp.py` calls `component_model.calc_causal_importances(...)`
  after the target forward.
- PGD utilities recompute target activations and CI by calling
  `model(..., cache_type="input")` and then `model.calc_causal_importances(...)`.
- Metrics call `model.components[path].apply_decomposed(...)` directly for
  attention-pattern reconstruction.
- Plotting, harvesting, app, and editing code freely read `components[...]`,
  `V`, `U`, `target_weight`, and CI outputs. Those can remain supported for
  offline/unwrapped models, but they should not be part of the FSDP training
  path.

The redesigned `ComponentModel.forward()` should take a structured request and
return a structured result. For example:

```python
result = component_model(
    batch,
    request=TrainingRequest(
        loss_configs=...,
        sampling=...,
        n_mask_samples=...,
        use_delta_component=...,
        ppgd_sources=...,
        collect_logs=True,
    ),
)
total_loss = result.total_loss
```

Internally, that single forward would:

1. Run the target-only forward and collect pre-weight activations.
2. Run the CI function from inside the same root forward.
3. Build masks for configured loss terms.
4. Run masked/decomposed forwards needed by those losses.
5. Compute reconstruction, importance/minimality, faithfulness, and PGD losses.
6. Return loss tensors and detached logging values.

The training loop would become mostly orchestration:

1. Build/update external non-parameter state such as PPGD sources and learning
   rates.
2. Call `component_model(batch, request=...)`.
3. Take PPGD source grads from returned losses if needed.
4. Backprop `result.total_loss`.
5. Clip and step optimizer.

Persistent PGD sources do not need to become model parameters. They can stay
outside the module and be passed as forward inputs. The important part is that
the forward computing the PGD loss is the same FSDP-rooted forward that uses
model params.

The delta-component path should be made site-local. Instead of passing a full
`weight_delta` tensor into each site, pass only a scalar/per-token delta mask.
Inside `DecomposedLinear.forward`, while `V`, `U`, and the frozen target weight
are already gathered, compute:

```text
component_acts = x @ V
masked_out = (component_acts * component_mask) @ U + bias
full_component_out = component_acts @ U + bias
target_out = frozen_linear(x)
delta_out = target_out - full_component_out
out = masked_out + delta_mask * delta_out
```

That preserves the "extra delta component" semantics without materializing
`W_target - V @ U` as a dense replicated tensor. It also keeps gradients through
`V` and `U` for the delta path.

Faithfulness loss should also be site-local. The simple implementation is for
each decomposed site to compute its own delta-norm scalar while it is executing
its normal forward and while its parameters are gathered. A collector object can
be bound to sites for the duration of `ComponentModel.forward()`, similar to the
current cache binding. If a configured site might not execute for a given
request, either require a target pass that touches all configured sites or add an
FSDP-safe parameter-regularizer forward for sites.

Direct site utilities need new homes:

- Training/eval losses should call `ComponentModel.forward(request=...)`, not
  `components[...]`.
- Analysis tools can operate on CPU/full checkpoints or explicitly enter an
  FSDP full-param/state-dict context.
- Specialized operations such as `apply_decomposed()` should either become
  request modes or registered FSDP forward methods, not arbitrary method calls on
  sharded modules.

With that design, a `ComponentModel` root becomes sensible, but it should still
be mostly a coordinator root:

- child FSDP units own large decomposed sites
- child FSDP units own CI transformer blocks
- large leftover params such as embeddings or `lm_head` are either separate
  child units or deliberately left replicated/ignored
- the root itself owns little or nothing large
- use explicit `reshard_after_forward=True` where peak memory matters

The main upside is conceptual: the whole differentiable decomposition step has a
single FSDP entry point, so parameter materialization happens under one module
tree and one set of FSDP ordering rules. The main cost is refactoring: losses,
metrics, PGD, and evaluation have to stop treating `ComponentModel` as a public
bag of modules.

## Should `target_model` Be The Root?

This is more natural for the target/decomposition path.

`ComponentModel.forward()` always invokes the target through `_run_batch`, so a
target-model root would actually enclose the target sites during target-only and
masked/decomposed forwards:

```text
ComponentModel                     not FSDP
  FSDP(target_model)               target-path root
    FSDP(decomposed site)          child, reshards after forward
    FSDP(decomposed site)          child, reshards after forward

  FSDP(GlobalSharedTransformerCiFn) CI-path root
    FSDP(CI TransformerBlock)       child
```

This splits the world into two sensible FSDP roots:

- target/decomposition root
- CI-function root

The target root should probably start as a coordinator/mostly-empty root:

- keep small leftover frozen target params replicated at first
- keep `DecomposedLinear` / `DecomposedEmbedding` as child FSDP units
- consider separate explicit FSDP units for very large leftover embeddings or
  `lm_head` only if replicated memory becomes the bottleneck

Wrapping target transformer blocks is another practical middle ground. If each
target block is an FSDP parent above its decomposed sites, then the large site
params become children and reshard after forward, while the block root only owns
small norms or undecomposed leftovers. A target-model coordinator root can still
sit above those block roots if a single target-path FSDP context is useful.

## Most Scalable Direction

The scalable target design should optimize peak memory first and communication
second:

1. Make large parameter groups child FSDP units with
   `reshard_after_forward=True`.
   The large groups are decomposed sites and CI transformer blocks.
2. Add a target-path FSDP parent, preferably `target_model` or target blocks, so
   decomposed sites are not independent roots.
3. Keep the CI transformer as its own root with block children. Set explicit
   resharding for the global CI root if the input/output projectors become large
   enough to matter.
4. Do not make `ComponentModel` a non-empty root unless all params it owns are
   always used through `ComponentModel.forward()`. Today that is not true.
5. Keep full-delta materialization out of the training reconstruction path.
   The previous `calc_weight_deltas_full()` path materialized full replicated
   deltas on every rank and detached them.
6. Keep delta-component math inside each decomposed site. The site already has
   `V`, `U`, and target weight gathered during its forward, so it can compute
   delta outputs locally without creating dense full `W_target - V @ U` tensors.
7. Either disable `FaithfulnessLoss` under FSDP or implement it as a
   differentiable site-local computation executed through FSDP-managed forwards.
8. Reduce the number of full model forwards per optimizer step where possible.
   The current loss stack can call the target/decomposed model many times before
   one backward, and FSDP pays gather cost on each call.
9. Keep activation memory separate in the mental model. FSDP shards params,
   grads, and optimizer state, but activations remain per rank. Target and CI
   gradient checkpointing are still required for large local batches/sequence
   lengths.
10. For multi-node scaling, consider a 2D/HSDP mesh later: shard within a fast
    node-local group and replicate across nodes if global all-gathers become the
    throughput bottleneck.

## Recommended Implementation Order

1. Minimal memory fix: pass explicit `reshard_after_forward=True` for the current
   large FSDP units. This does not solve all coordination/prefetch issues, but it
   avoids relying on root-default behavior.
2. Replace faithfulness with a site-local implementation if it is needed under
   FSDP. The reconstruction delta path is already site-local; faithfulness is
   the remaining full-delta training loss.
3. Add a target-path root. Start with target blocks or an ignored-param
   `target_model` coordinator root, then test target-only, masked, PPGD, and eval
   forwards.
4. Only after that, experiment with sharding leftover embeddings/`lm_head`.
   Treat that as a separate memory/communication tradeoff, not as a prerequisite
   for making decomposed sites behave correctly.
5. Implement FSDP-compatible checkpoint save/load before using FSDP for long
   production runs.
