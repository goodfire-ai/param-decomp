"""Vendored JAX TMS (Toy Model of Superposition) target — the first non-LM
`DecomposedModel`, with `leading_axes=()` (no position axes; the waist is `[B, n_features]`).

Torch reference (read-only ground truth): `param_decomp_lab/experiments/tms/models.py`.
The target is `out = relu(linear2(linear1(x)))`: `linear1` `(n_features -> n_hidden)` no
bias, `linear2` `(n_hidden -> n_features)` with bias, weights TIED
(`linear2.weight = linear1.weight.T`). The DECOMPOSITION is UNTIED — `linear1` and
`linear2` are two sites with independent `(V, U)` (the TMS PD configs set
`decomposition_targets: [linear1, linear2]`, `tied_weights: null`).

There is no prefix: the whole model is decomposed, so the "residual" entering the
decomposed part is the raw input `x` `[B, n_features]` and `tms_input_residual` is the
identity. The recon comparison is MSE on the post-ReLU `[B, n_features]` output (NOT
KL): `recon_loss_fn = tms_mse` (torch `recon_loss_mse`, mean over batch × features).

Site weights are right-mult oriented like the LM targets (`site_out = x @ W.T`): for
`linear1` `W` is `(n_hidden, n_features)`; for `linear2` `(n_features, n_hidden)`.
`n_hidden_layers > 0` is refused — the production TMS configs (5-2, 40-10) use none."""

from dataclasses import dataclass
from typing import Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float

from jax_single_pool.ci_fn import CIValues
from jax_single_pool.llama8b import DecompVU
from jax_single_pool.lm import DecomposedModel, SiteC, SiteSpec

LINEAR1 = "linear1"
LINEAR2 = "linear2"
SITE_NAMES = (LINEAR1, LINEAR2)


class CIFnCallable(Protocol):
    """The CI-fn surface the TMS target-CI probe needs: `__call__(site_inputs) -> CIValues`
    (satisfied by both `ci_fn.CIFn` and `ci_fn_mlp.LayerwiseMLPCIFn`)."""

    def __call__(self, site_inputs: dict[str, Array]) -> CIValues: ...


@dataclass(frozen=True)
class TMSConfig:
    n_features: int
    n_hidden: int


class TMSTarget(eqx.Module):
    """Frozen TMS weights, right-mult oriented. `W1` `(n_hidden, n_features)`,
    `W2` `(n_features, n_hidden)`, `b2` `(n_features,)`."""

    W1: Float[Array, "n_hidden n_features"]
    W2: Float[Array, "n_features n_hidden"]
    b2: Float[Array, " n_features"]


def site_dims(cfg: TMSConfig, name: str) -> tuple[int, int]:
    """(d_in, d_out) right-mult orientation."""
    match name:
        case "linear1":
            return cfg.n_features, cfg.n_hidden
        case "linear2":
            return cfg.n_hidden, cfg.n_features
        case _:
            raise AssertionError(f"unknown TMS site {name!r}")


def canonical_site_cs(site_cs: tuple[SiteC, ...]) -> tuple[SiteC, ...]:
    """Canonical order is `linear1` then `linear2` (= computation order). Both sites must
    be present exactly once."""
    names = [s.name for s in site_cs]
    assert sorted(names) == sorted(SITE_NAMES), f"TMS sites must be {SITE_NAMES}, got {names}"
    by_name = {s.name: s for s in site_cs}
    return tuple(by_name[name] for name in SITE_NAMES)


def site_specs(cfg: TMSConfig, site_cs: tuple[SiteC, ...]) -> tuple[SiteSpec, ...]:
    site_cs = canonical_site_cs(site_cs)
    specs = []
    for site in site_cs:
        assert site.C >= 1, site
        d_in, d_out = site_dims(cfg, site.name)
        specs.append(SiteSpec(site.name, d_in, d_out, site.C))
    return tuple(specs)


def _frozen_site_weight(target: TMSTarget, name: str) -> Array:
    match name:
        case "linear1":
            return target.W1
        case "linear2":
            return target.W2
        case _:
            raise AssertionError(f"unknown TMS site {name!r}")


def clean_output(target: TMSTarget, resid: Float[Array, "B n_features"]) -> Array:
    """The all-frozen forward — the recon target (SPEC S3). `resid` is the raw input `x`."""
    hidden = resid @ target.W1.T
    return jax.nn.relu(hidden @ target.W2.T + target.b2)


def site_inputs(target: TMSTarget, resid: Float[Array, "B n_features"]) -> dict[str, Array]:
    """Clean CI inputs per site (SPEC S4): `linear1` reads the input `x`; `linear2` reads
    the frozen `linear1(x)`."""
    return {LINEAR1: resid, LINEAR2: resid @ target.W1.T}


def _site_out(
    x: Array,
    V: Array,
    U: Array,
    W: Array,
    mask: Array | None,
    delta_mask: Array | None,
    route: Array | None,
) -> Array:
    """One decomposed linear (SPEC §4.1), `llama8b._site_out` for the positionless waist:
    `((x@V)*m)@U + (x@Δ)*d`, routed per cell against frozen `x @ W.T`."""
    acts = x @ V
    if mask is not None:
        acts = acts * mask
    out = acts @ U
    if delta_mask is not None:
        delta = W - (V @ U).T  # (d_out, d_in)
        out = out + delta_mask[..., None] * (x @ delta.T)
    if route is not None:
        out = jnp.where(route[..., None], out, x @ W.T)
    return out


def _masked_site_out(
    components: DecompVU,
    site: str,
    W: Array,
    x_in: Array,
    masks: dict[str, Array],
    delta_masks: dict[str, Array],
    routes: dict[str, Array] | None,
    live_set: frozenset[str],
    has_delta: bool,
    collect: dict[str, Array] | None,
) -> Array:
    if site not in live_set:
        return x_in @ W.T
    V, U = components.site(site)
    out = _site_out(
        x_in, V, U, W, masks[site], delta_masks[site] if has_delta else None,
        None if routes is None else routes[site],
    )  # fmt: skip
    if collect is not None:
        collect[site] = out
    return out


def _run_masked(
    target: TMSTarget,
    components: DecompVU,
    resid: Float[Array, "B n_features"],
    masks: dict[str, Array],
    delta_masks: dict[str, Array],
    routes: dict[str, Array] | None,
    live: tuple[str, ...],
    has_delta: bool,
    collect: dict[str, Array] | None,
) -> Array:
    """The masked decomposed forward (SPEC §4.1, S2): sites in `live` run their decomposed
    forward; the rest run the frozen `x @ W.T` path. The `linear2` site reads the
    (possibly masked) `linear1` output so a masked linear1 propagates."""
    live_set = frozenset(live)
    site_args = (masks, delta_masks, routes, live_set, has_delta, collect)
    hidden = _masked_site_out(components, LINEAR1, target.W1, resid, *site_args)
    pre_relu = _masked_site_out(components, LINEAR2, target.W2, hidden, *site_args) + target.b2
    return jax.nn.relu(pre_relu)


def masked_output(
    target: TMSTarget,
    components: DecompVU,
    resid: Float[Array, "B n_features"],
    masks: dict[str, Array],
    delta_masks: dict[str, Array],
    routes: dict[str, Array] | None,
    live: tuple[str, ...],
    has_delta: bool,
) -> Array:
    return _run_masked(target, components, resid, masks, delta_masks, routes, live, has_delta, None)


def masked_site_outputs(
    target: TMSTarget,
    components: DecompVU,
    resid: Float[Array, "B n_features"],
    masks: dict[str, Array],
    delta_masks: dict[str, Array],
    routes: dict[str, Array] | None,
    live: tuple[str, ...],
    has_delta: bool,
) -> dict[str, Array]:
    """Per-`live`-site decomposed output of the masked forward (SPEC S31)."""
    collect: dict[str, Array] = {}
    _run_masked(target, components, resid, masks, delta_masks, routes, live, has_delta, collect)
    assert set(collect) == set(live), (sorted(collect), sorted(live))
    return collect


def weight_deltas_fp32(
    target: TMSTarget, components: DecompVU, sites: tuple[SiteSpec, ...]
) -> dict[str, Array]:
    """fp32 `W − V@U` per site from fp32 masters (SPEC N2; faithfulness input)."""
    out: dict[str, Array] = {}
    for spec in sites:
        W = _frozen_site_weight(target, spec.name)
        V, U = components.site(spec.name)
        out[spec.name] = W.astype(jnp.float32) - (V.astype(jnp.float32) @ U.astype(jnp.float32)).T
    return out


def tms_mse(masked: Float[Array, "B n_features"], clean: Float[Array, "B n_features"]) -> Array:
    """MSE recon over the post-ReLU output, mean over batch × features (torch
    `recon_loss_mse`: `sum((pred-target)^2) / pred.numel()`), fp32."""
    masked = masked.astype(jnp.float32)
    clean = clean.astype(jnp.float32)
    return jnp.mean((masked - clean) ** 2)


def tms_decomposed_model(cfg: TMSConfig, sites: tuple[SiteSpec, ...]) -> DecomposedModel:
    sites = site_specs(cfg, tuple(SiteC(s.name, s.C) for s in sites))
    return DecomposedModel(
        sites=sites,
        leading_axes=(),
        clean_output=clean_output,
        site_inputs=site_inputs,
        masked_output=masked_output,
        masked_site_outputs=masked_site_outputs,
        weight_deltas=lambda target, components: weight_deltas_fp32(target, components, sites),
        recon_loss_fn=tms_mse,
    )


def replicate_target(target: TMSTarget, mesh: Mesh) -> TMSTarget:
    """Replicate the tiny frozen TMS weights on every device (the `replicate_frozen`
    analog; V/U / CI placement reuses the generic plan)."""
    repl = NamedSharding(mesh, P())
    return jax.tree.map(lambda a: jax.device_put(a, repl) if eqx.is_array(a) else a, target)


def tms_input_residual(prefix: None, inputs: Float[Array, "B n_features"]) -> Array:
    """No prefix: the residual entering the decomposed model IS the raw input `x`. Kept
    as a `prefix_residual_fn` so the generic `run.py` harvest path is uniform across
    targets (`prefix` is unused / `None`)."""
    del prefix
    return inputs


# ----------------------------- from-scratch pretraining -----------------------------


class _TMSTrainable(eqx.Module):
    """The pretrain-trainable subset: `W1` and `b2` (`W2 = W1.T` is the tie, never a
    separate leaf). An eqx Module so optax sees clean per-array leaves."""

    W1: Float[Array, "n_hidden n_features"]
    b2: Float[Array, " n_features"]


def init_tms_target(cfg: TMSConfig, key: Array) -> TMSTarget:
    """Untrained TMS target: `W1` Kaiming-uniform like torch `nn.Linear`, `b2 = 0`,
    weights TIED at init (`W2 = W1.T`)."""
    bound = 1.0 / cfg.n_features**0.5
    W1 = jax.random.uniform(key, (cfg.n_hidden, cfg.n_features), minval=-bound, maxval=bound)
    return TMSTarget(W1=W1, W2=W1.T, b2=jnp.zeros((cfg.n_features,)))


def sample_sparse_features(
    key: Array,
    batch: int,
    n_features: int,
    feature_probability: float,
    generation_type: str,
) -> Float[Array, "B n_features"]:
    """Synthetic sparse-feature batch (torch `SparseFeatureDataset`): each feature
    activates with `U[0,1]` value, gated by `feature_probability` (`at_least_zero_active`)
    or by exactly one active feature per row (`exactly_one_active`)."""
    value_key, gate_key = jax.random.split(key)
    values = jax.random.uniform(value_key, (batch, n_features))
    match generation_type:
        case "at_least_zero_active":
            mask = jax.random.uniform(gate_key, (batch, n_features)) < feature_probability
            return values * mask
        case "exactly_one_active":
            active = jax.random.randint(gate_key, (batch,), 0, n_features)
            one_hot = jax.nn.one_hot(active, n_features)
            return values * one_hot
        case _:
            raise AssertionError(f"unsupported TMS generation type {generation_type!r}")


def pretrain_tms_target(
    cfg: TMSConfig,
    feature_probability: float,
    generation_type: str,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> TMSTarget:
    """From-scratch pretrain of the frozen TMS target (the Anthropic TMS objective
    `mean((|x| - relu_out)^2)`, importance = 1). The target is kept TIED throughout
    (`W2 = W1.T`): only `W1` and `b2` train, matching the torch `TMSModel.tie_weights_()`
    contract. Returns the trained target with `W2 = W1.T`.

    Deterministic in `seed`: this replaces the missing wandb pretrain checkpoint so a TMS
    PD run is reproducible from the config alone."""
    import optax

    key = jax.random.PRNGKey(seed)
    init_key, data_key = jax.random.split(key)
    target = init_tms_target(cfg, init_key)
    # Only W1 and b2 train; W2 is the tie W1.T, reconstructed each loss eval. eqx Modules
    # are pytrees, so optax + eqx.apply_updates keep the (W1, b2) array types honest.
    trainable = _TMSTrainable(W1=target.W1, b2=target.b2)
    optimizer = optax.adamw(lr, weight_decay=0.0)
    opt_state = optimizer.init(eqx.filter(trainable, eqx.is_array))

    @jax.jit
    def step(
        trainable: "_TMSTrainable", opt_state: optax.OptState, x: Array
    ) -> tuple["_TMSTrainable", optax.OptState]:
        def loss_fn(trainable: "_TMSTrainable") -> Array:
            hidden = x @ trainable.W1.T
            out = jax.nn.relu(hidden @ trainable.W1 + trainable.b2)  # W2 = W1.T
            return jnp.mean((jnp.abs(x) - out) ** 2)

        grad = eqx.filter_grad(loss_fn)(trainable)
        updates, opt_state = optimizer.update(grad, opt_state, eqx.filter(trainable, eqx.is_array))
        return eqx.apply_updates(trainable, updates), opt_state

    for s in range(steps):
        x = sample_sparse_features(
            jax.random.fold_in(data_key, s), batch_size, cfg.n_features,
            feature_probability, generation_type,
        )  # fmt: skip
        trainable, opt_state = step(trainable, opt_state, x)
    return TMSTarget(W1=trainable.W1, W2=trainable.W1.T, b2=trainable.b2)


# ----------------------------- ground-truth target-CI eval -----------------------------


def identity_ci_error(ci_vals: Float[Array, "n_features C"], tolerance: float) -> int:
    """Discrete identity-CI distance (torch `IdentityCIPattern.distance_from`): permute
    columns toward identity (Hungarian on `-ci`), then over the `min(shape)` square block
    count off-diagonal entries `> tolerance` plus on-diagonal entries `< 1 - tolerance`.

    `ci_vals` is the `lower_leaky` CI of the single-feature probe (one row per feature)."""
    from scipy.optimize import linear_sum_assignment

    ci = np.asarray(ci_vals, dtype=np.float64)
    assert ci.ndim == 2, ci.shape
    n_features, C = ci.shape
    size = min(n_features, C)
    _, col_indices = linear_sum_assignment(-ci[:size])
    assigned = list(col_indices)
    remaining = [c for c in range(C) if c not in set(assigned)]
    perm = np.array(assigned + remaining, dtype=np.int64)
    ci = ci[:, perm]

    block = ci[:size, :size]
    off_diag_mask = ~np.eye(size, dtype=bool)
    off_diag_errors = int((block[off_diag_mask] > tolerance).sum())
    on_diag_errors = int((np.diagonal(block) < (1 - tolerance)).sum())
    return off_diag_errors + on_diag_errors


SINGLE_FEATURE_PROBE_MAGNITUDE = 0.75
"""Torch `IdentityCIError.input_magnitude` — the single-active-feature probe value."""


def single_feature_probe(n_features: int) -> Float[Array, "n_features n_features"]:
    """The single-feature probe (torch `get_single_feature_causal_importances`):
    `eye(n_features) * 0.75`, one active feature per row."""
    return jnp.eye(n_features) * SINGLE_FEATURE_PROBE_MAGNITUDE


def single_feature_ci(
    lm: DecomposedModel,
    target: TMSTarget,
    ci_fn: "CIFnCallable",
    n_features: int,
) -> dict[str, Array]:
    """Feed the single-feature probe and read the `lower_leaky` CI per site,
    `{site: [n_features, C]}`."""
    return ci_fn(lm.site_inputs(target, single_feature_probe(n_features))).lower
