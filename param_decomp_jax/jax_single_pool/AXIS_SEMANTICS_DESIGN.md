# Axis semantics: the generic `[*leading, d]` waist

This documents the activation-waist contract after generalizing the trainer core from a
fixed `(B, T)` (batch, sequence) waist to a generic `[*leading, d]` waist. It is the
companion to `SPEC.md` (which is normative on the algorithm) — this doc is normative on
the *shape contract* the generic loop and the per-domain targets agree on.

## The waist

Per decomposed **site**, the trainer threads two shape families through the step:

- **activations** `[*leading, d]` — the residual entering the suffix, per-site inputs
  (`[*leading, d_in]`), per-site outputs (`[*leading, d_out]`), final model output
  (`[*leading, vocab]` for an LM).
- **masks / CI** `[*leading, C]`, **delta masks / routes** `[*leading]`.

`leading = (batch,) + named position axes`. The **batch axis is ever-present and
semantics-free**: it is the data/shard axis (GSPMD pins it to `P('dp', …)`), nothing in
the method treats one batch element differently from another. The named position axes are
domain-specific: an LM has one (`sequence`); a TMS-style target has none.

### CI is independent over every leading axis

A single team decision underpins the whole simplification: **CI is always independent
over every leading axis** (the old `broadcast_ci` knob is dropped). The CI fn emits one
logit per `(*leading, component)` cell; there is no axis over which CI is shared or
pooled. Consequently there is **no per-axis CI semantics** — only axis *names*. The core
never needs to know what a leading axis *means*; it only needs its *extent* (for sampling
shapes) and the invariant that every per-site tensor in a forward shares the same
`*leading` prefix.

This is why the generic loop can stay free of domain branches: it reads
`leading = residual.shape[:-1]` once and uses it uniformly for routing-draw shapes,
stochastic source shapes, delta-mask shapes, and the imp-min / KL reductions
(`math.prod(shape[:-1])`, `axis=tuple(range(ndim-1))`).

## Axis names live on the model and the CI fn

- **`DecomposedModel.leading_axes: tuple[str, ...]`** names the position axes (batch is
  implicit): `("sequence",)` for an LM, `()` for TMS.
- **`CIFn.expects_axes: tuple[str, ...]`** names the position axes the CI fn is built for.
  The LM CI fn applies RoPE over a single `sequence` axis, so it declares
  `("sequence",)`.

At trainer construction (`run_state.init_train_state`) we **assert
`ci_fn.expects_axes == model.leading_axes`** and fail early. This lets the CI fn stay
per-domain (it can encode position structure however it likes — RoPE here) without the
generic core ever adapting: the core only ever sees the opaque `*leading` prefix, and the
construction-time check guarantees the two halves agree on how many position axes there
are and what they are.

## What changed in the core (all per-domain branches stay out of the loop)

The generalization is mechanical — every site that assumed exactly two leading axes now
reads an opaque `leading`:

- `train.py` (keystone): `batch, seq = residual.shape[0], residual.shape[1]` →
  `leading = residual.shape[:-1]`, threaded to the routing samplers, fresh-PGD source
  init, and stochastic delta-mask shapes (the `(batch, seq)` tuple param is now
  `leading_shape: tuple[int, ...]`).
- `losses.py`: imp-min and `kl_per_position` reductions — `shape[0]*shape[1]` →
  `math.prod(shape[:-1])`, `axis=(0,1)` → `axis=tuple(range(ndim-1))`.
- `recon.py`: `RoutingSampler` and the three sampler bodies — `tuple[int, int]` →
  `tuple[int, ...]` (bodies were already generic over the shape tuple).
- `adversary.py`: `init_persistent_sources` takes a `leading_shape` directly;
  `init_fresh_pgd_sources` takes the model's `leading` and spells `scope` (`c`/`bc`/`bsc`)
  over it. `source_masks` already broadcast via `…[..., :-1]` ellipsis.
- `llama8b_sharding.py`: `init_sources_sharded` builds the per-scope `leading_shape`
  (`sc` → `(1, T)`, `bsc` → `(B, T)`) for the LM before placing.
- `eval.py` / `slow_eval.py` / `hidden_acts_eval.py`: the same `shape[:-1]` / `prod`
  generalizations for delta-mask shapes and position counts.
- `lm.py` type aliases: `Float[Array, "B T C"]` → `Float[Array, "*leading C"]`, etc.

The LM `CIFn` internals (`b, t, d = x.shape`, RoPE over `t`) are **per-domain and stay
exactly as written** — they are the concrete realization of `expects_axes=("sequence",)`,
not core. Likewise the concrete Llama target forwards keep their honest `"b t d"`
annotations.

## The shape contract is enforced (jaxtyping + beartype)

The waist invariant — *all per-site tensors in one forward share one `*leading` prefix* —
is enforced at trace time, not just documented. `@jaxtyped(typechecker=beartype)`
decorates the generic `masked_forward` (residual `[*leading d]`, masks `[*leading _]`,
delta masks `[*leading]`, routes `[*leading]` — `*leading` is bound consistently across
all of them by jaxtyping), the core `step` (residual `[*leading d]`), and the pure loss
fns (`kl_per_position`, `faithfulness_loss`, `importance_minimality_terms`). The check
runs on shaped JAX tracers under `jit`, so a target that emits a ragged leading prefix
fails at trace time with a shape error rather than silently miscomputing a reduction.

Per-site `C` / `d_in` / `d_out` are deliberately **anonymous** (`_`) inside the per-site
dict annotations — they vary across sites, so they must not be bound across dict entries;
only the `*leading` prefix is shared and bound.

## Designed extensions (no current consumer — NOT implemented here)

These fall out of the same axis-name machinery and are recorded so the next change has a
target, but nothing consumes them today:

- **PPGD source scope as "the subset of leading axes the source spans."** Today's
  `c`/`sc`/`bsc` are LM-specific spellings (collapse batch, collapse nothing, etc.).
  Generalized, a scope is a subset of `leading_axes` (plus the implicit batch axis) that
  the adversarial source is *independent over*; the complementary axes are singleton and
  broadcast. The current LM code keeps the `c`/`sc`/`bsc` literals; generalizing scope to
  arbitrary axis subsets is out of scope for this change.
- **Tying as a shared `vu` leaf.** Tied decomposed weights would be expressed as a single
  shared leaf in the `vu` pytree referenced by multiple sites, rather than a separate
  tying mechanism. No current consumer.

## Out of scope for this change

- **TMS** is not ported here (separate task). `leading_axes=()` is the contract a TMS
  target would declare, but no TMS `DecomposedModel` exists in this distribution yet.
- **Source-scope over arbitrary axes** (above) — `c`/`sc`/`bsc` stay LM-shaped.
