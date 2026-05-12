# FSDP weight-delta computation — problem, opportunity, suggestion

*Branch `feature/fsdp-wrap`. Context for whoever owns the FSDP wrap going forward.*

---

## The problem

Each training step we call `calc_weight_deltas_full` once
(`param_decomp/utils/fsdp.py:29`). That function:

1. Calls `.unshard()` on every `DecomposedSite` — all-gathers `V`, `U`, *and*
   `target.weight` to full tensors on every rank.
2. Computes `target_weight − V @ U` per site. Every rank does the full `V @ U`
   matmul redundantly and materializes a `(d_out, d_in)` dense delta.
3. Calls `.detach()` and then `.full_tensor()` to convert from DTensor →
   regular Tensor.
4. Reshards.
5. Returns a dict `{site_name: dense_delta}` that's threaded into the loss
   pipeline as `weight_deltas`.

That dict is the dominant peak-memory artifact at the 4B-target scale: every
rank carries one full target-shaped tensor per site, *in addition to* the
gathered `V`/`U`/`W`.

There are also two problems beyond the cost:

- **Correctness regression** — the `.detach()` at step 3 is load-bearing for
  avoiding a DTensor leak into downstream loss math, but it severs the
  autograd path from the faithfulness loss back into `V`/`U`. The comment at
  `param_decomp/run_param_decomp.py:346-349` flags this explicitly:
  *"faithfulness backward won't flow into V/U (known correctness regression
  under FSDP — proper fix is to compute delta-norms inside each site's
  forward)."* So under FSDP, faithfulness currently doesn't train the
  decomposition.
- **It undoes the architectural win of fused sites.** `decomposed_module.py`'s
  header explicitly motivates fusing `V`/`U` with the target submodule so
  they live in the *same* FSDP unit, dodging the "hook crosses FSDP units"
  problem. Precomputing the delta dict outside any forward and feeding it
  back in re-introduces exactly that pattern — we gather, materialize,
  reshard, and then the consumer's forward gathers `V`/`U` again to use the
  precomputed delta.

---

## The opportunity

There are two unrelated consumers of `weight_deltas`, and the dict-of-full-
deltas API conflates them:

**(A) `faithfulness_loss`** (`param_decomp/metrics/faithfulness_loss.py:14`)
is just `sum(delta**2)` over every element of every site — a pure scalar
reduction. It never indexes into the delta. Cost should be one local
sum-of-squares per shard plus one scalar `all_reduce(SUM)` per step. Memory
cost should be zero `(d_out, d_in)` allocations.

**(B) `use_delta_component` consumers** (PPGD, stochastic recon, etc.) feed
the delta into `_decomposed_forward` at `decomposed_module.py:189`:

```python
einsum(x, weight_delta, "... d_in, d_out d_in -> ... d_out")
```

That einsum happens *inside* an FSDP-wrapped site forward, where
`V`/`U`/`target.weight` are already gathered by the site's pre-forward hook.
Precomputing the delta outside is double work.

So the opportunity is to delete `calc_weight_deltas_full` entirely and split
its two responsibilities: a sharded scalar reduction for (A), and lazy
in-forward delta materialization for (B).

---

## Suggested design

### (A) Faithfulness as a sharded reduction

Pick a common shard axis for `W`, `V`, `U` so each rank can compute

```python
local_sum_sq = ((W_local - (V @ U)_local) ** 2).sum()
```

and the metric does a single scalar `all_reduce(SUM)` for `sum_sq` and for
`total_params`.

FSDP2's default `Shard(dim=0)` gives:

- `W: (d_out, d_in)`  sharded on `d_out`
- `U: (C, d_out)`     sharded on `C`           ← misaligned
- `V: (d_in, C)`      sharded on `d_in`        ← misaligned

The natural fix is to declare placements explicitly at site construction via
`distribute_tensor`:

- replicate `V` (typically the smaller param: `d_in × C` vs `C × d_out`
  when `d_out = 4·d_model`),
- shard `U` along `d_out` (its dim 1),
- leave `W` row-sharded on `d_out`.

Then `V @ U_local` produces the local `d_out` slice of `V @ U`, aligned with
`W_local`, and the rest is element-wise.

If keeping `V` replicated is too expensive for some site (e.g. the embed
matrix with vocab >> d_model), the fallback is to keep `V` sharded on dim 1
(its `C` axis) along with `U` sharded on dim 0 (also `C`), compute a partial
sum along `C`, and `DTensor.redistribute` into `Shard(d_out)` before
subtracting from `W_local`. But the replicated-V version is the simple
default and almost certainly fine — V is small in every layer except possibly
the embedding.

Cost per step: **one scalar allreduce per site** instead of three
all-gathers per site. Memory: **no `(d_out, d_in)` allocation anywhere**.

This also fixes the correctness regression: the reduction stays in DTensor
land, autograd flows back into the sharded `V`/`U` through the allreduce, and
the `.detach()` workaround disappears.

### (B) Lazy delta inside `_decomposed_forward`

Stop precomputing the delta outside the forward. Change the
`WeightDeltaAndMask` payload to carry only the `weight_delta_mask`. Inside
`_decomposed_forward`, when a mask is present, form

```python
delta = self.target_weight - self.component_weight
```

right there. At that point V/U/W are already gathered by FSDP's per-site
pre-forward hook, so no extra comms, no extra peak memory beyond what the
forward already needs, and the autograd graph stays connected.

After (A) and (B):

- `calc_weight_deltas_full` can be deleted.
- The `weight_deltas: dict[str, Tensor]` arg disappears from
  `compute_losses`, PPGD, stochastic recon, etc.
- It's replaced by a `use_delta_component: bool` flag threaded through to
  the sites (most call sites already gate on `use_delta_component` — they
  just stop materializing the dict).

### Suggested order

1. **Land (A) first.** It's a self-contained refactor of `FaithfulnessLoss`
   plus a small DTensor-placement change at site construction. Verify the
   faithfulness value matches the current (broken-gradient) number, then
   verify that V/U gradients from faithfulness are nonzero under FSDP — i.e.
   you've actually fixed the correctness regression flagged in
   `run_param_decomp.py:346-349`.
2. **Land (B) as a follow-up.** Bigger surface area because every metric
   that takes `weight_deltas` changes signature, but each call site is
   mechanical: delete the kwarg, set a bool.
3. **Delete** `calc_weight_deltas_full` and the `weight_deltas` arg from
   `compute_losses` / PPGD.
