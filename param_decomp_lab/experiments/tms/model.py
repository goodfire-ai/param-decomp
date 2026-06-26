"""Vendored JAX TMS (Toy Model of Superposition) target — the first non-LM
`DecomposedModel`, with `leading_axes=()` (no position axes; the waist is `[B, n_features]`).

Torch reference (read-only ground truth): `param_decomp_lab/experiments/tms/models.py`.
The target is `out = relu(linear2(hidden_layers(linear1(x))))`: `linear1`
`(n_features -> n_hidden)` no bias, optional FROZEN `hidden_layers.{i}` `(n_hidden ->
n_hidden)` no bias, `linear2` `(n_hidden -> n_features)` with bias, with `linear1`/`linear2`
weights TIED (`linear2.weight = linear1.weight.T`). The DECOMPOSITION is UNTIED — each site
gets its own independent `(V, U)`.

There is no prefix: the whole model is decomposed, so the "residual" entering the
decomposed part is the raw input `x` `[B, n_features]` and `tms_input_residual` is the
identity. The recon comparison is MSE on the post-ReLU `[B, n_features]` output (NOT
KL): `recon_loss_fn = tms_mse` (torch `recon_loss_mse`, mean over batch × features).

Site weights are right-mult oriented like the LM targets (`site_out = x @ W.T`): for
`linear1` `W` is `(n_hidden, n_features)`; for `hidden_layers.{i}` `(n_hidden, n_hidden)`;
for `linear2` `(n_features, n_hidden)`. The frozen `hidden_layers.{i}` give PD an extra
dense decomposition target (torch's `-id` configs); they are initialized to identity or
frozen-random per `TMSConfig.hidden_layer_init` and never trained.
"""

from dataclasses import dataclass
from typing import Literal, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float

from param_decomp.ci_fn import CI
from param_decomp.components import DecompVU, SiteC, SiteSpec, site_out

LINEAR1 = "linear1"
LINEAR2 = "linear2"

HiddenLayerInit = Literal["identity", "random"]


def hidden_layer_name(i: int) -> str:
    return f"hidden_layers.{i}"


def site_names_for(n_hidden_layers: int) -> tuple[str, ...]:
    """Canonical computation order: `linear1`, the `hidden_layers.{i}` in index order,
    then `linear2`."""
    hidden = tuple(hidden_layer_name(i) for i in range(n_hidden_layers))
    return (LINEAR1, *hidden, LINEAR2)


class CIFnCallable(Protocol):
    """The CI-fn surface the TMS target-CI probe needs: `__call__(taps) -> CI` plus the
    `input_names` the probe feeds (satisfied by `ci_fn.LayerwiseMLPCIFn` / `GlobalMLPCIFn`)."""

    input_names: tuple[str, ...]

    def __call__(self, taps: dict[str, Array], *, remat: bool) -> CI: ...


@dataclass(frozen=True)
class TMSConfig:
    n_features: int
    n_hidden: int
    n_hidden_layers: int = 0
    hidden_layer_init: HiddenLayerInit = "identity"
    init_bias_to_zero: bool = True


@dataclass(frozen=True)
class TMSTargetConfig:
    """The lab TMS target config carried on `ExperimentConfig.target` (satisfies the core
    `TargetSites` protocol via `sites`). Pretrained from scratch in-process from `pretrain_*`
    (no weight artifact). `global_batch` is the PD-step batch (the toy has no parquet
    `DataConfig`, so its batch lives here). `hidden_layers.{i}` sites (the torch `-id`
    variant) are FROZEN at init and threaded through as extra dense decomposition targets."""

    n_features: int
    n_hidden: int
    sites: tuple[SiteC, ...]
    pretrain_steps: int
    pretrain_batch_size: int
    pretrain_lr: float
    pretrain_seed: int
    feature_probability: float
    data_generation_type: str
    global_batch: int
    n_hidden_layers: int = 0
    hidden_layer_init: HiddenLayerInit = "identity"
    init_bias_to_zero: bool = True


class TMSTarget(eqx.Module):
    """Frozen TMS weights, right-mult oriented. `W1` `(n_hidden, n_features)`,
    `hidden` an ordered tuple of FROZEN `(n_hidden, n_hidden)` layers, `W2`
    `(n_features, n_hidden)`, `b2` `(n_features,)`."""

    W1: Float[Array, "n_hidden n_features"]
    hidden: tuple[Float[Array, "n_hidden n_hidden"], ...]
    W2: Float[Array, "n_features n_hidden"]
    b2: Float[Array, " n_features"]


def _parse_hidden_layer_index(name: str) -> int | None:
    """The `i` for a `hidden_layers.{i}` site, else `None`."""
    prefix = "hidden_layers."
    if not name.startswith(prefix):
        return None
    return int(name[len(prefix) :])


def site_dims(cfg: TMSConfig, name: str) -> tuple[int, int]:
    """(d_in, d_out) right-mult orientation."""
    if name == LINEAR1:
        return cfg.n_features, cfg.n_hidden
    if name == LINEAR2:
        return cfg.n_hidden, cfg.n_features
    i = _parse_hidden_layer_index(name)
    assert i is not None and 0 <= i < cfg.n_hidden_layers, f"unknown TMS site {name!r}"
    return cfg.n_hidden, cfg.n_hidden


def canonical_site_cs(site_cs: tuple[SiteC, ...]) -> tuple[SiteC, ...]:
    """Canonical order is `linear1`, the `hidden_layers.{i}` in index order, then `linear2`
    (= computation order). The site set is inferred from the configured names (the number
    of hidden-layer sites = the number of `hidden_layers.{i}` entries); each must appear
    exactly once."""
    by_name = {s.name: s for s in site_cs}
    assert len(by_name) == len(site_cs), f"duplicate TMS site in {[s.name for s in site_cs]}"
    n_hidden_layers = sum(1 for s in site_cs if _parse_hidden_layer_index(s.name) is not None)
    expected = site_names_for(n_hidden_layers)
    assert sorted(by_name) == sorted(expected), (
        f"TMS sites must be {expected}, got {sorted(by_name)}"
    )
    return tuple(by_name[name] for name in expected)


def site_specs(cfg: TMSConfig, site_cs: tuple[SiteC, ...]) -> tuple[SiteSpec, ...]:
    site_cs = canonical_site_cs(site_cs)
    n_hidden_layers = sum(1 for s in site_cs if _parse_hidden_layer_index(s.name) is not None)
    assert n_hidden_layers == cfg.n_hidden_layers, (
        f"config n_hidden_layers={cfg.n_hidden_layers} but {n_hidden_layers} hidden-layer sites"
    )
    specs = []
    for site in site_cs:
        assert site.C >= 1, site
        d_in, d_out = site_dims(cfg, site.name)
        specs.append(SiteSpec(site.name, d_in, d_out, site.C))
    return tuple(specs)


def _frozen_site_weight(target: TMSTarget, name: str) -> Array:
    if name == LINEAR1:
        return target.W1
    if name == LINEAR2:
        return target.W2
    i = _parse_hidden_layer_index(name)
    assert i is not None and 0 <= i < len(target.hidden), f"unknown TMS site {name!r}"
    return target.hidden[i]


def _frozen_hidden_forward(target: TMSTarget, hidden: Array) -> Array:
    """Thread the activation through the frozen `hidden_layers.{i}` (right-mult)."""
    for W in target.hidden:
        hidden = hidden @ W.T
    return hidden


def clean_output(target: TMSTarget, resid: Float[Array, "B n_features"]) -> Array:
    """The all-frozen forward — the recon target (SPEC S3). `resid` is the raw input `x`."""
    hidden = resid @ target.W1.T
    hidden = _frozen_hidden_forward(target, hidden)
    return jax.nn.relu(hidden @ target.W2.T + target.b2)


def site_inputs(target: TMSTarget, resid: Float[Array, "B n_features"]) -> dict[str, Array]:
    """Clean CI inputs per site (SPEC S4): each site reads the frozen output of the chain
    up to it — `linear1` reads `x`, `hidden_layers.{i}` reads the frozen output through
    `hidden_layers.{i-1}`, `linear2` reads the frozen output through the last hidden layer."""
    inputs: dict[str, Array] = {LINEAR1: resid}
    hidden = resid @ target.W1.T
    for i, W in enumerate(target.hidden):
        inputs[hidden_layer_name(i)] = hidden
        hidden = hidden @ W.T
    inputs[LINEAR2] = hidden
    return inputs


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
    out = site_out(
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
    forward; the rest run the frozen `x @ W.T` path. Each downstream site reads the
    (possibly masked) output of the site before it, so a masked site propagates."""
    live_set = frozenset(live)
    site_args = (masks, delta_masks, routes, live_set, has_delta, collect)
    hidden = _masked_site_out(components, LINEAR1, target.W1, resid, *site_args)
    for i, W in enumerate(target.hidden):
        hidden = _masked_site_out(components, hidden_layer_name(i), W, hidden, *site_args)
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


class TMSDecomposedModel(eqx.Module):
    """The TMS `DecomposedModel` (the `lm.py` contract; SPEC §1), positionless
    (`leading_axes=()`).

    Carries the FROZEN `TMSTarget` weights as a field — threaded into the jitted step as a
    pytree arg, weights traced not baked. The TRAINABLE V/U (`vu: DecompVU`) is an explicit
    method arg, NOT a field (separate lifecycle). `sites` / `leading_axes` are static."""

    target: TMSTarget
    sites: tuple[SiteSpec, ...] = eqx.field(static=True)
    leading_axes: tuple[str, ...] = eqx.field(static=True)

    @property
    def site_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.sites)

    def shardings(self, mesh: Mesh) -> "TMSDecomposedModel":
        """Replicate every frozen leaf on the `dp` mesh — TMS weights are tiny."""
        repl = NamedSharding(mesh, P())
        return jax.tree.map(lambda _a: repl, self)

    @staticmethod
    def recon_loss_fn(
        masked_output: Float[Array, "B n_features"], clean_output: Float[Array, "B n_features"]
    ) -> Array:
        return tms_mse(masked_output, clean_output)

    def clean_output(self, resid: Float[Array, "B n_features"]) -> Array:
        return clean_output(self.target, resid)

    def read_activations(
        self, resid: Float[Array, "B n_features"], wanted: tuple[str, ...]
    ) -> dict[str, Array]:
        inputs = site_inputs(self.target, resid)
        return {k: inputs[k] for k in wanted}

    def masked_output(
        self,
        vu: DecompVU,
        resid: Float[Array, "B n_features"],
        masks: dict[str, Array],
        delta_masks: dict[str, Array],
        routes: dict[str, Array] | None,
        live: tuple[str, ...],
        has_delta: bool,
        *,
        remat: bool,
    ) -> Array:
        def forward(
            vu: DecompVU,
            resid: Array,
            masks: dict[str, Array],
            delta_masks: dict[str, Array],
            routes: dict[str, Array] | None,
        ) -> Array:
            return masked_output(
                self.target, vu, resid, masks, delta_masks, routes, live, has_delta
            )

        forward = jax.checkpoint(forward) if remat else forward
        return forward(vu, resid, masks, delta_masks, routes)

    def masked_site_outputs(
        self,
        vu: DecompVU,
        resid: Float[Array, "B n_features"],
        masks: dict[str, Array],
        delta_masks: dict[str, Array],
        routes: dict[str, Array] | None,
        live: tuple[str, ...],
        has_delta: bool,
    ) -> dict[str, Array]:
        return masked_site_outputs(
            self.target, vu, resid, masks, delta_masks, routes, live, has_delta
        )

    def weight_deltas(self, vu: DecompVU) -> dict[str, Array]:
        return weight_deltas_fp32(self.target, vu, self.sites)


def tms_decomposed_model(
    cfg: TMSConfig, target: TMSTarget, sites: tuple[SiteSpec, ...]
) -> TMSDecomposedModel:
    """Wrap a pretrained `TMSTarget` + decomposition config into the `DecomposedModel`."""
    sites = site_specs(cfg, tuple(SiteC(s.name, s.C) for s in sites))
    return TMSDecomposedModel(target=target, sites=sites, leading_axes=())


def replicate_target[T: (TMSTarget, TMSDecomposedModel)](target: T, mesh: Mesh) -> T:
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
    separate leaf; the `hidden_layers.{i}` are FROZEN at init). An eqx Module so optax
    sees clean per-array leaves."""

    W1: Float[Array, "n_hidden n_features"]
    b2: Float[Array, " n_features"]


def _init_hidden_layers(
    cfg: TMSConfig, key: Array
) -> tuple[Float[Array, "n_hidden n_hidden"], ...]:
    """Frozen `hidden_layers.{i}`: identity (the `-id` configs) or frozen standard-normal
    (torch `fixed_random_hidden_layers`). Right-mult oriented `(n_hidden, n_hidden)`; for
    identity the orientation is moot (`I.T == I`)."""
    match cfg.hidden_layer_init:
        case "identity":
            return tuple(jnp.eye(cfg.n_hidden) for _ in range(cfg.n_hidden_layers))
        case "random":
            keys = jax.random.split(key, cfg.n_hidden_layers)
            return tuple(jax.random.normal(k, (cfg.n_hidden, cfg.n_hidden)) for k in keys)


def init_tms_target(cfg: TMSConfig, key: Array) -> TMSTarget:
    """Untrained TMS target: `W1` Kaiming-uniform like torch `nn.Linear`, `b2` zero when
    `init_bias_to_zero` (else `nn.Linear`'s `uniform(-1/sqrt(fan), 1/sqrt(fan))` bias init),
    `linear1`/`linear2` TIED (`W2 = W1.T`), `hidden_layers.{i}` frozen per `hidden_layer_init`."""
    w_key, hidden_key, bias_key = jax.random.split(key, 3)
    bound = 1.0 / cfg.n_features**0.5
    W1 = jax.random.uniform(w_key, (cfg.n_hidden, cfg.n_features), minval=-bound, maxval=bound)
    if cfg.init_bias_to_zero:
        b2 = jnp.zeros((cfg.n_features,))
    else:
        bias_bound = 1.0 / cfg.n_hidden**0.5
        b2 = jax.random.uniform(bias_key, (cfg.n_features,), minval=-bias_bound, maxval=bias_bound)
    return TMSTarget(W1=W1, hidden=_init_hidden_layers(cfg, hidden_key), W2=W1.T, b2=b2)


_EXACTLY_N_ACTIVE: dict[str, int] = {
    "exactly_one_active": 1,
    "exactly_two_active": 2,
    "exactly_three_active": 3,
    "exactly_four_active": 4,
    "exactly_five_active": 5,
}


def sample_sparse_features(
    key: Array,
    batch: int,
    n_features: int,
    feature_probability: float,
    generation_type: str,
) -> Float[Array, "B n_features"]:
    """Synthetic sparse-feature batch (torch `SparseFeatureDataset`, `value_range=(0,1)`):
    `at_least_zero_active` gates each feature independently by `feature_probability`;
    `exactly_n_active` (`exactly_one_active` … `exactly_five_active`) activates exactly `n`
    distinct features per row via a random permutation. Inactive features are 0.

    NOTE: torch's `synced_inputs` (co-activating feature groups) and the no-zero-sample
    rejection variant (`_generate_multi_feature_batch_no_zero_samples`) are not ported —
    no JAX config uses them; add them if a config needs them."""
    value_key, gate_key = jax.random.split(key)
    values = jax.random.uniform(value_key, (batch, n_features))
    if generation_type == "at_least_zero_active":
        mask = jax.random.uniform(gate_key, (batch, n_features)) < feature_probability
        return values * mask
    n = _EXACTLY_N_ACTIVE.get(generation_type)
    assert n is not None, f"unsupported TMS generation type {generation_type!r}"
    assert n <= n_features, f"cannot activate {n} of {n_features} features"
    sort_key = jax.random.uniform(gate_key, (batch, n_features))
    active = jnp.argsort(sort_key, axis=-1)[:, :n]  # n distinct features per row
    one_hot = jax.nn.one_hot(active, n_features).sum(axis=-2)  # [B, n_features], n ones per row
    return values * one_hot


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
    contract; the `hidden_layers.{i}` stay FROZEN at their init. Returns the trained
    target with `W2 = W1.T`.

    Deterministic in `seed`: this replaces the missing wandb pretrain checkpoint so a TMS
    PD run is reproducible from the config alone."""
    import optax

    key = jax.random.PRNGKey(seed)
    init_key, data_key = jax.random.split(key)
    target = init_tms_target(cfg, init_key)
    hidden = target.hidden  # frozen; closed over by the loss, never updated
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
            h = x @ trainable.W1.T
            for W in hidden:
                h = h @ W.T
            out = jax.nn.relu(h @ trainable.W1 + trainable.b2)  # W2 = W1.T
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
    return TMSTarget(W1=trainable.W1, hidden=hidden, W2=trainable.W1.T, b2=trainable.b2)


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


def dense_ci_error(
    ci_vals: Float[Array, "B C"], k: int, tolerance: float, min_entries: int = 1
) -> int:
    """Discrete dense-CI distance (torch `DenseCIPattern.distance_from`): sort columns by
    total mass (densest first), then over the first `k` columns count one error per
    column with fewer than `min_entries` strong activations (`>= 1 - tolerance`), and over
    the remaining columns count one error per weak activation (`> tolerance`, should be
    inactive).

    Used for the `-id` variant's frozen `hidden_layers.{i}` site (`k = n_hidden`): a dense
    layer should keep ALL its `n_hidden` directions live."""
    ci = np.asarray(ci_vals, dtype=np.float64)
    assert ci.ndim == 2, ci.shape
    C = ci.shape[1]
    assert k <= C, f"expected at least {k} columns, got {C}"

    column_mass = ci.sum(axis=0)
    perm = np.argsort(-column_mass)
    ci = ci[:, perm]

    strong_per_column = (ci >= 1 - tolerance).sum(axis=0)
    missing_strong = np.clip(min_entries - strong_per_column, a_min=0, a_max=None)
    first_k_error = int(missing_strong[:k].sum())

    weak_per_column = (ci > tolerance).sum(axis=0)
    inactive_error = int(weak_per_column[k:].sum())
    return first_k_error + inactive_error


SINGLE_FEATURE_PROBE_MAGNITUDE = 0.75
"""Torch `IdentityCIError.input_magnitude` — the single-active-feature probe value."""


def single_feature_probe(n_features: int) -> Float[Array, "n_features n_features"]:
    """The single-feature probe (torch `get_single_feature_causal_importances`):
    `eye(n_features) * 0.75`, one active feature per row."""
    return jnp.eye(n_features) * SINGLE_FEATURE_PROBE_MAGNITUDE


def single_feature_ci(
    lm: TMSDecomposedModel,
    ci_fn: "CIFnCallable",
    n_features: int,
) -> dict[str, Array]:
    """Feed the single-feature probe and read the `lower_leaky` CI per site,
    `{site: [n_features, C]}`."""
    probe = single_feature_probe(n_features)
    return ci_fn(lm.read_activations(probe, ci_fn.input_names), remat=False).lower


# ----------------------------- visualizations -----------------------------


def plot_intro_diagram(target: TMSTarget, filepath: str) -> None:
    """The Anthropic 2-D TMS polygon plot (only valid for `n_hidden == 2`): each feature is
    a point `W1.T[f]` in the 2-D hidden space plus a segment from the origin. Writes a PNG."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    WA = np.asarray(target.W1.T)  # [n_features, 2]
    assert WA.shape[1] == 2, f"intro diagram needs n_hidden==2, got {WA.shape[1]}"

    fig, ax = plt.subplots(figsize=(4, 4), facecolor="#FCFBF8")
    ax.set_facecolor("#FCFBF8")
    segments = [[(0.0, 0.0), (row[0], row[1])] for row in WA]
    ax.add_collection(LineCollection(segments, colors="#444444", linewidths=1.0))
    ax.scatter(WA[:, 0], WA[:, 1], c="#7A1F2B", s=30, zorder=3)

    ax.set_aspect("equal")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    for side in ("left", "bottom"):
        ax.spines[side].set_position("center")
    for side in ("right", "top"):
        ax.spines[side].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cosine_similarity_distribution(target: TMSTarget, filepath: str) -> None:
    """Scatter the off-diagonal pairwise cosine similarities of the unit-normalized feature
    directions `W1.T[f]` on a 1-D strip in `[-1, 1]`, with a reference line at the
    superposition-floor `1 / sqrt(n_hidden)`. Writes a PNG."""
    import matplotlib.pyplot as plt

    WA = np.asarray(target.W1.T)  # [n_features, n_hidden]
    n_features, n_hidden = WA.shape
    rows = WA / (np.linalg.norm(WA, axis=1, keepdims=True) + 1e-12)
    cosine_sims = rows @ rows.T
    off_diag = cosine_sims[~np.eye(n_features, dtype=bool)]

    fig, ax = plt.subplots(figsize=(6, 2), facecolor="#FCFBF8")
    ax.set_facecolor("#FCFBF8")
    jitter = np.random.default_rng(0).uniform(-0.3, 0.3, size=off_diag.shape)
    ax.scatter(off_diag, jitter, c="#7A1F2B", s=12, alpha=0.6)
    ax.axvline(1.0 / n_hidden**0.5, color="#444444", linestyle="--", linewidth=1.0)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.0, 1.0)
    ax.set_yticks([])
    ax.set_xlabel("cosine similarity")

    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
