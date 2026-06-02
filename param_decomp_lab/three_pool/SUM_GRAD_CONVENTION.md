# SUM-grad convention (proposal)

A structural redesign of the 3-pool gradient-assembly scaling, replacing the
per-instance "pre-scale to survive a downstream reduction" patches (4 recurring
bugs, latest = PR #545's PPGD `×n_ci`) with a single convention.

## The bug class

The CI-fn weights and the V/U weights are each REPLICATED across ranks. Their
gradients are assembled from multiple producers (stoch, faith, imp-min, ppgd)
and reduced across ranks. Every recurring bug had the same shape: a producer
pre-scaled its gradient by a *pool-size factor* (`n_ci`, `n_per_block`) so its
contribution would survive a downstream AVG-reduce it couldn't locally see. That
factor leaks pool-size knowledge into the gradient VALUES, and a single
differentiated scalar that feeds two destinations with different reductions
(stoch → CI leaves ÷n_ci AND V/U ÷n_per_block) is guaranteed wrong on a
non-square topology.

## The convention

**Every gradient crossing a cross-rank reduction is a partial SUM, normalized
only by the honest GLOBAL count (global examples/positions × sites), carrying NO
pool-size transport factor. All data-parallel gradient reductions are SUM.**

Partial sums compose: `SUM(partials) = total`. So no producer needs to know any
pool's size; the only normalization is the honest global count, which is locally
derivable (`P_global = n_positions_local × n_per_block`, `n_examples_global =
n_examples_local × n_ppgd`). The conversion factor that turns a local count into
a global count is NOT a transport factor — it is part of computing the honest
denominator, and it disappears entirely on a square topology only by coincidence.

### Consequences

1. **The grad all-reduce is SUM.** After an *all*-reduce every rank holds the
   identical value either way; under SUM that value is the TOTAL, which equals
   the single-pool gradient *because each producer already divided by the global
   count*. The optimizer steps on it directly.
2. **`cross_pool_clip_grad_norm(n_replicas)` is UNCHANGED.** Subtle: this divide
   is about counting DISTINCT parameters for the global norm, not about the grad
   reduce. After the in-pool *all*-reduce (SUM or AVG), every replica holds the
   IDENTICAL grad; the pool-wide sq-SUM therefore counts each block's params
   `n_per_block` times either way, so the `n_replicas` dedup stays. (The grad
   VALUE differs — SUM gives the single-pool total, AVG gave total/n_per_block —
   but the replica COUNT being summed is the same.)
3. **stoch's one scale feeds both destinations.** CI leaves (→ CI pool, SUM) and
   V/U (→ LW block, SUM) both want the same partial-sum scale
   `coeff_stoch / (P_global × n_sites_total)`. The double-duty bug is structurally
   impossible now: there is only one correct scale and it serves both.
4. **PPGD's `×n_ci` DIES.** V/U and CI both want `coeff_ppgd / n_examples_global`;
   the CI path no longer needs the extra `×n_ci` to survive an AVG. The two
   collapse to one scale — the shape the V/U path (which never had a bug) always
   had.

## The wrinkle: replicated contributions

The convention is clean for genuine DP partials (disjoint batch slices). It does
NOT, by itself, handle REPLICATED contributions — gradients that are IDENTICAL on
every rank in the reduction group because they were computed from replicated
inputs rather than a disjoint data slice:

- **faith V/U** (`_faithfulness_loss`): computed from the replicated V/U weights →
  identical on every block rank → under SUM, `n_per_block×` too big.
- **broadcast PPGD V/U**: sum-reduced within PPGD then broadcast to all block
  ranks → identical on every block rank → same `n_per_block×` problem.
- **imp-min CI**: the autograd-aware `dist_fn.all_reduce(SUM)` backward
  SUM-reduces the *replicated* upstream gradient across the CI pool, leaving each
  rank with `n_ci×` its true partial. Under the old AVG this was exactly the
  factor that made it correct; under SUM it is `n_ci×` too big.

Three ways to handle each:

  (a) **Divide the replicated contribution by the replica count before the SUM.**
      Rejected: this REINTRODUCES the pool-size factor into a producer — exactly
      what the convention abolishes. It only relocates the factor.
  (b) **Contribute once.** Compute the replicated contribution on a single rank
      (the block leader) so there is no replica to undo. Chosen for faith and
      broadcast PPGD V/U.
  (c) **Detached-global-residual.** Make the forward value global but the backward
      flow only through the local contribution:
      `S = local + (all_reduce_sum(local.detach()) - local.detach())`.
      Forward `S = global_sum`; backward `∂S/∂local = 1`, no cross-rank term, so
      each rank gets its TRUE partial which SUM-composes. Chosen for imp-min
      (its loss genuinely needs the global sum inside the `log2`, so option (b)
      doesn't apply — it isn't a replica, it's a global reduction).

### faith / broadcast PPGD → contribute once (option b)

- **faith**: run the faith backward on the **block leader only**. The leader's
  `.grad` then carries the full single-pool faith grad once; non-leaders
  contribute zero faith. After the block SUM every rank holds it exactly once.
  Faith is already divided by `numel_global`, so the leader's value is already
  the single-pool grad — no further scaling.
- **broadcast PPGD V/U**: skip the in-block broadcast; the block **leader** adds
  the received PPGD grad to its `.grad`, non-leaders add nothing. After the block
  SUM every rank holds it once.

These two changes mean the block all-reduce SUM now combines ONLY:
`leader_faith + leader_ppgd + Σ_ranks stoch_partial_r` = the single-pool total.

### imp-min → detached-global-residual (option c)

`_importance_minimality_loss` replaces the autograd-aware `dist_fn.all_reduce`
with the detached-global-residual on `per_component_sums`. Forward identical
(global sum inside `finalize_imp_min`'s `log2` and mean), backward flows only
through this rank's local CI values → a true partial → SUM-composes under the CI
pool's SUM all-reduce. The `×n_ci` knowledge leaves the imp path entirely.

## The honest verdict

Does the SUM convention ELIMINATE pool-size knowledge from producers?

- **From the data-parallel producers: YES.** stoch, ppgd V/U, ppgd CI all lose
  every `n_ci` / `n_per_block` *transport* factor. The #545 `×n_ci` is deleted.
  stoch's two destinations collapse to one scale. The remaining `n_per_block` in
  stoch's denom is not a transport factor — it is the `local→global` position
  count conversion, which any honest global normalization needs.
- **From the replicated contributions: NO — but it RELOCATES the count to a
  structurally honest place.** faith and broadcast-PPGD no longer *scale* by
  `n_per_block`; instead they *contribute once* (a topology fact: "this grad is
  replicated, emit it on one rank"). imp-min no longer *relies on AVG to cancel*
  `n_ci`; instead it *states* "my backward is a local partial" via the residual
  trick. The replica count does not appear as a numeric factor in any producer's
  gradient value — it appears as a *placement* decision (which rank emits) or a
  *graph* decision (detach the cross-rank term).

Net: the convention is **not a free win** — replicated contributions still need
the system to know they are replicated. But it converts an error-prone numeric
coupling ("multiply by the size of a pool you can't see, to survive a reduce that
happens elsewhere") into a local, inspectable structural statement ("this is a
partial; emit it once" / "this is replicated; detach the global term"). That is a
genuine simplification for the DP majority and a clearer, harder-to-get-wrong
encoding for the replicated minority — not a lateral move.
