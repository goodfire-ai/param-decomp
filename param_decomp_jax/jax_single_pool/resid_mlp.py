"""Vendored JAX ResidualMLP target — the fourth `DecomposedModel` and the second
non-LM bundle, positionless (`leading_axes=()`; the waist is the residual stream
`[B, d_embed]`).

Torch reference (read-only ground truth): `param_decomp_lab/experiments/resid_mlp/`
(`models.py` the architecture, `train_resid_mlp.py` the read-off pretrain objective,
`data.py` the synthetic sparse features). The target is the SPD/APD residual-stream toy:

    resid = x @ W_E
    for layer in layers:          # n_layers MLP blocks reading/writing the residual stream
        h = act(resid @ W_inᵀ + b_in)
        resid = resid + (h @ W_outᵀ + b_out)
    out = resid @ W_U

`W_E` `(n_features, d_embed)` and `W_U` `(d_embed, n_features)` are FIXED (the canonical
toy uses a random unit-norm embedding with `W_U = W_Eᵀ`); the per-layer MLP matrices
train. The DECOMPOSITION targets the MLP matrices: sites `layers.{i}.mlp_in`
`(d_embed → d_mlp)` and `layers.{i}.mlp_out` `(d_mlp → d_embed)`, each an UNTIED
`(V, U)` (`tied_weights: null`).

`W_E` is the PREFIX: the residual entering the decomposed part is `x @ W_E` and
`resid_mlp_input_residual` maps `(W_E, x) -> x @ W_E`. The recon comparison is MSE on the
model OUTPUT `[B, n_features]` (NOT KL): `recon_loss_fn = resid_mlp_mse`.

Site weights are right-mult oriented like the LM targets (`site_out = x @ Wᵀ`): for
`mlp_in` `W` is `(d_mlp, d_embed)`; for `mlp_out` `(d_embed, d_mlp)`."""

from collections.abc import Callable
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
from jax_single_pool.llama8b import DecompVU, _site_out
from jax_single_pool.lm import DecomposedModel, SiteC, SiteSpec

MLP_IN = "mlp_in"
MLP_OUT = "mlp_out"
KINDS = (MLP_IN, MLP_OUT)


class CIFnCallable(Protocol):
    """The CI-fn surface the ResidMLP target-CI probe needs: `__call__(site_inputs) ->
    CIValues` (satisfied by both `ci_fn.CIFn` and `ci_fn_mlp.LayerwiseMLPCIFn`)."""

    def __call__(self, site_inputs: dict[str, Array]) -> CIValues: ...


@dataclass(frozen=True)
class ResidMLPConfig:
    n_features: int
    d_embed: int
    d_mlp: int
    n_layers: int
    act_fn_name: str
    in_bias: bool
    out_bias: bool
    fixed_identity_embedding: bool
    """`W_E = I` (requires `n_features == d_embed`): the residual stream IS the feature
    basis, the unambiguous clean ground-truth regime (torch `fixed_identity_embedding`).
    False uses a fixed random unit-norm embedding (torch `fixed_random_embedding`)."""


def _act_fn(name: str) -> Callable[[Array], Array]:
    match name:
        case "gelu":
            return lambda x: jax.nn.gelu(x, approximate=False)
        case "relu":
            return jax.nn.relu
        case _:
            raise AssertionError(f"unknown ResidMLP act fn {name!r}")


class ResidMLPLayer(eqx.Module):
    """One MLP block, right-mult oriented. `W_in` `(d_mlp, d_embed)`, `W_out`
    `(d_embed, d_mlp)`; biases are `None` when the config disables them."""

    W_in: Float[Array, "d_mlp d_embed"]
    W_out: Float[Array, "d_embed d_mlp"]
    b_in: Float[Array, " d_mlp"] | None
    b_out: Float[Array, " d_embed"] | None


class ResidMLPTarget(eqx.Module):
    """Frozen ResidualMLP weights: fixed `W_E` / `W_U` embeddings and the per-layer MLP
    blocks (ordered by layer index)."""

    W_E: Float[Array, "n_features d_embed"]
    W_U: Float[Array, "d_embed n_features"]
    layers: tuple[ResidMLPLayer, ...]
    act_fn_name: str = eqx.field(static=True)


def parse_site_name(name: str) -> tuple[int, str]:
    """`layers.{i}.{mlp_in,mlp_out}` -> (layer, kind); rejects anything else."""
    parts = name.split(".")
    assert len(parts) == 3 and parts[0] == "layers" and parts[2] in KINDS, (
        f"unsupported ResidMLP site name {name!r}"
    )
    return int(parts[1]), parts[2]


def site_name(layer: int, kind: str) -> str:
    assert kind in KINDS, kind
    return f"layers.{layer}.{kind}"


def canonical_site_cs(site_cs: tuple[SiteC, ...]) -> tuple[SiteC, ...]:
    """Canonical site order: layer-ascending, `mlp_in` before `mlp_out` within a layer
    (= computation order). Names must parse and be unique."""
    names = [s.name for s in site_cs]
    assert len(set(names)) == len(names), f"duplicate sites in {names}"

    def order_key(site: SiteC) -> tuple[int, int]:
        layer, kind = parse_site_name(site.name)
        return layer, KINDS.index(kind)

    return tuple(sorted(site_cs, key=order_key))


def site_dims(cfg: ResidMLPConfig, kind: str) -> tuple[int, int]:
    """(d_in, d_out) right-mult orientation."""
    match kind:
        case "mlp_in":
            return cfg.d_embed, cfg.d_mlp
        case "mlp_out":
            return cfg.d_mlp, cfg.d_embed
        case _:
            raise AssertionError(f"unknown ResidMLP kind {kind!r}")


def site_specs(cfg: ResidMLPConfig, site_cs: tuple[SiteC, ...]) -> tuple[SiteSpec, ...]:
    """Shape-resolved specs in canonical order. Every layer must contribute BOTH its
    `mlp_in` and `mlp_out` site (the masked forward threads the residual through both)."""
    site_cs = canonical_site_cs(site_cs)
    expected = {site_name(layer, kind) for layer in range(cfg.n_layers) for kind in KINDS}
    got = {s.name for s in site_cs}
    assert got == expected, f"ResidMLP sites must be exactly {sorted(expected)}, got {sorted(got)}"
    specs = []
    for site in site_cs:
        assert site.C >= 1, site
        _, kind = parse_site_name(site.name)
        specs.append(SiteSpec(site.name, *site_dims(cfg, kind), site.C))
    return tuple(specs)


def _frozen_site_weight(target: ResidMLPTarget, name: str) -> Array:
    layer, kind = parse_site_name(name)
    block = target.layers[layer]
    match kind:
        case "mlp_in":
            return block.W_in
        case "mlp_out":
            return block.W_out
        case _:
            raise AssertionError(f"unknown ResidMLP kind {kind!r}")


def clean_output(target: ResidMLPTarget, resid: Float[Array, "B d_embed"]) -> Array:
    """The all-frozen forward — the recon target (SPEC S3). `resid` is `x @ W_E`."""
    act = _act_fn(target.act_fn_name)
    for block in target.layers:
        h = resid @ block.W_in.T
        if block.b_in is not None:
            h = h + block.b_in
        out = act(h) @ block.W_out.T
        if block.b_out is not None:
            out = out + block.b_out
        resid = resid + out
    return resid @ target.W_U


def site_inputs(target: ResidMLPTarget, resid: Float[Array, "B d_embed"]) -> dict[str, Array]:
    """Clean CI inputs per site (SPEC S4): `mlp_in` reads the clean residual entering its
    layer; `mlp_out` reads the clean post-activation hidden of its layer."""
    act = _act_fn(target.act_fn_name)
    inputs: dict[str, Array] = {}
    for layer, block in enumerate(target.layers):
        inputs[site_name(layer, MLP_IN)] = resid
        pre = resid @ block.W_in.T
        if block.b_in is not None:
            pre = pre + block.b_in
        hidden = act(pre)
        inputs[site_name(layer, MLP_OUT)] = hidden
        out = hidden @ block.W_out.T
        if block.b_out is not None:
            out = out + block.b_out
        resid = resid + out
    return inputs


def _decomposed_or_frozen(
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
    """One site's output: the decomposed `_site_out` if live, else the frozen `x @ Wᵀ`."""
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
    target: ResidMLPTarget,
    components: DecompVU,
    resid: Float[Array, "B d_embed"],
    masks: dict[str, Array],
    delta_masks: dict[str, Array],
    routes: dict[str, Array] | None,
    live: tuple[str, ...],
    has_delta: bool,
    collect: dict[str, Array] | None,
) -> Array:
    """The masked decomposed forward (SPEC §4.1, S2): each layer's `mlp_in`/`mlp_out` runs
    its decomposed forward if live, else the frozen path. The biases are added AFTER the
    site output (they live outside the decomposed matrix), the residual accumulates as in
    the frozen forward, and the readout `resid @ W_U` is frozen."""
    act = _act_fn(target.act_fn_name)
    site_args = (masks, delta_masks, routes, frozenset(live), has_delta, collect)
    for layer, block in enumerate(target.layers):
        pre = _decomposed_or_frozen(
            components, site_name(layer, MLP_IN), block.W_in, resid, *site_args
        )
        if block.b_in is not None:
            pre = pre + block.b_in
        hidden = act(pre)
        out = _decomposed_or_frozen(
            components, site_name(layer, MLP_OUT), block.W_out, hidden, *site_args
        )
        if block.b_out is not None:
            out = out + block.b_out
        resid = resid + out
    return resid @ target.W_U


def masked_output(
    target: ResidMLPTarget,
    components: DecompVU,
    resid: Float[Array, "B d_embed"],
    masks: dict[str, Array],
    delta_masks: dict[str, Array],
    routes: dict[str, Array] | None,
    live: tuple[str, ...],
    has_delta: bool,
) -> Array:
    return _run_masked(target, components, resid, masks, delta_masks, routes, live, has_delta, None)


def masked_site_outputs(
    target: ResidMLPTarget,
    components: DecompVU,
    resid: Float[Array, "B d_embed"],
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
    target: ResidMLPTarget, components: DecompVU, sites: tuple[SiteSpec, ...]
) -> dict[str, Array]:
    """fp32 `W − (V@U)ᵀ` per site from fp32 masters (SPEC N2; faithfulness input)."""
    out: dict[str, Array] = {}
    for spec in sites:
        W = _frozen_site_weight(target, spec.name)
        V, U = components.site(spec.name)
        out[spec.name] = W.astype(jnp.float32) - (V.astype(jnp.float32) @ U.astype(jnp.float32)).T
    return out


def resid_mlp_mse(
    masked: Float[Array, "B n_features"], clean: Float[Array, "B n_features"]
) -> Array:
    """MSE recon over the model output, mean over batch × features (fp32)."""
    masked = masked.astype(jnp.float32)
    clean = clean.astype(jnp.float32)
    return jnp.mean((masked - clean) ** 2)


def resid_mlp_decomposed_model(cfg: ResidMLPConfig, sites: tuple[SiteSpec, ...]) -> DecomposedModel:
    sites = site_specs(cfg, tuple(SiteC(s.name, s.C) for s in sites))
    return DecomposedModel(
        sites=sites,
        leading_axes=(),
        clean_output=clean_output,
        site_inputs=site_inputs,
        masked_output=masked_output,
        masked_site_outputs=masked_site_outputs,
        weight_deltas=lambda target, components: weight_deltas_fp32(target, components, sites),
        recon_loss_fn=resid_mlp_mse,
    )


def replicate_target(target: ResidMLPTarget, mesh: Mesh) -> ResidMLPTarget:
    """Replicate the tiny frozen ResidMLP weights on every device (V/U / CI placement
    reuses the generic replicated plan, as for TMS)."""
    repl = NamedSharding(mesh, P())
    return jax.tree.map(lambda a: jax.device_put(a, repl) if eqx.is_array(a) else a, target)


def resid_mlp_input_residual(prefix: ResidMLPTarget, inputs: Float[Array, "B n_features"]) -> Array:
    """The residual entering the decomposed model is `x @ W_E`. The prefix IS the frozen
    target (it carries `W_E`); kept as a `prefix_residual_fn` so the generic harvest path
    is uniform across targets."""
    return inputs @ prefix.W_E


# ----------------------------- from-scratch pretraining -----------------------------


class _ResidMLPTrainable(eqx.Module):
    """The pretrain-trainable subset: the per-layer MLP blocks (the embeddings `W_E`/`W_U`
    are FIXED — `fixed_random_embedding` — so they never appear here)."""

    layers: tuple[ResidMLPLayer, ...]


def _init_layer(cfg: ResidMLPConfig, key: Array) -> ResidMLPLayer:
    """One MLP block, torch `nn.Linear` Kaiming-uniform init (`init_param_` fan-in
    uniform `bound = 1/√fan_in`), zero biases when enabled."""
    in_key, out_key = jax.random.split(key)
    in_bound = 1.0 / cfg.d_embed**0.5
    out_bound = 1.0 / cfg.d_mlp**0.5
    W_in = jax.random.uniform(in_key, (cfg.d_mlp, cfg.d_embed), minval=-in_bound, maxval=in_bound)
    W_out = jax.random.uniform(
        out_key, (cfg.d_embed, cfg.d_mlp), minval=-out_bound, maxval=out_bound
    )
    return ResidMLPLayer(
        W_in=W_in,
        W_out=W_out,
        b_in=jnp.zeros((cfg.d_mlp,)) if cfg.in_bias else None,
        b_out=jnp.zeros((cfg.d_embed,)) if cfg.out_bias else None,
    )


def _init_embedding(cfg: ResidMLPConfig, key: Array) -> Float[Array, "n_features d_embed"]:
    """`W_E = I` for the identity regime, else a fixed random unit-norm embedding."""
    if cfg.fixed_identity_embedding:
        assert cfg.n_features == cfg.d_embed, (cfg.n_features, cfg.d_embed)
        return jnp.eye(cfg.n_features)
    raw = jax.random.normal(key, (cfg.n_features, cfg.d_embed))
    return raw / jnp.linalg.norm(raw, axis=-1, keepdims=True)


def init_resid_mlp_target(cfg: ResidMLPConfig, key: Array) -> ResidMLPTarget:
    """Untrained ResidMLP target: a FIXED embedding (`W_U = W_Eᵀ`) and Kaiming-uniform MLP
    blocks. `fixed_identity_embedding` gives `W_E = I` (the residual stream IS the feature
    basis — the unambiguous clean ground-truth regime); otherwise a random unit-norm
    embedding."""
    embed_key, layers_key = jax.random.split(key)
    W_E = _init_embedding(cfg, embed_key)
    layer_keys = jax.random.split(layers_key, cfg.n_layers)
    layers = tuple(_init_layer(cfg, layer_keys[i]) for i in range(cfg.n_layers))
    return ResidMLPTarget(W_E=W_E, W_U=W_E.T, layers=layers, act_fn_name=cfg.act_fn_name)


def sample_sparse_features(
    key: Array,
    batch: int,
    n_features: int,
    feature_probability: float,
    generation_type: str,
) -> Float[Array, "B n_features"]:
    """Synthetic sparse-feature batch (torch `SparseFeatureDataset` with ResidMLP's
    `value_range=(-1, 1)`): each feature takes a `U[-1, 1]` value, gated by
    `feature_probability` (`at_least_zero_active`) or exactly one active per row
    (`exactly_one_active`)."""
    value_key, gate_key = jax.random.split(key)
    values = jax.random.uniform(value_key, (batch, n_features), minval=-1.0, maxval=1.0)
    match generation_type:
        case "at_least_zero_active":
            mask = jax.random.uniform(gate_key, (batch, n_features)) < feature_probability
            return values * mask
        case "exactly_one_active":
            active = jax.random.randint(gate_key, (batch,), 0, n_features)
            return values * jax.nn.one_hot(active, n_features)
        case _:
            raise AssertionError(f"unsupported ResidMLP generation type {generation_type!r}")


def readoff_labels(
    target: ResidMLPTarget, x: Float[Array, "B n_features"]
) -> Float[Array, "B n_features"]:
    """The read-off pretrain target `act_fn(coeffs·x) + x` with trivial unit coeffs (torch
    `calc_act_plus_resid_labels` + `use_trivial_label_coeffs`)."""
    return _act_fn(target.act_fn_name)(x) + x


def pretrain_resid_mlp_target(
    cfg: ResidMLPConfig,
    feature_probability: float,
    generation_type: str,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> ResidMLPTarget:
    """From-scratch pretrain of the frozen ResidMLP target (the read-off MSE objective
    `mean((out − (act_fn(x) + x))²)`, trivial unit label coeffs). The fixed embedding
    `W_E`/`W_U` is held constant; only the MLP blocks train.

    Deterministic in `seed`: this replaces the missing wandb pretrain checkpoint so a
    ResidMLP PD run is reproducible from the config alone."""
    import optax

    key = jax.random.PRNGKey(seed)
    init_key, data_key = jax.random.split(key)
    target = init_resid_mlp_target(cfg, init_key)
    trainable = _ResidMLPTrainable(layers=target.layers)
    optimizer = optax.adamw(lr, weight_decay=0.0)
    opt_state = optimizer.init(eqx.filter(trainable, eqx.is_array))

    @jax.jit
    def step(
        trainable: "_ResidMLPTrainable", opt_state: optax.OptState, x: Array
    ) -> tuple["_ResidMLPTrainable", optax.OptState]:
        def loss_fn(trainable: "_ResidMLPTrainable") -> Array:
            frozen = eqx.tree_at(lambda t: t.layers, target, trainable.layers)
            out = clean_output(frozen, x @ target.W_E)
            return jnp.mean((out - readoff_labels(target, x)) ** 2)

        grad = eqx.filter_grad(loss_fn)(trainable)
        updates, opt_state = optimizer.update(grad, opt_state, eqx.filter(trainable, eqx.is_array))
        return eqx.apply_updates(trainable, updates), opt_state

    for s in range(steps):
        x = sample_sparse_features(
            jax.random.fold_in(data_key, s), batch_size, cfg.n_features,
            feature_probability, generation_type,
        )  # fmt: skip
        trainable, opt_state = step(trainable, opt_state, x)
    return eqx.tree_at(lambda t: t.layers, target, trainable.layers)


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
"""The single-active-feature probe value (torch `IdentityCIError.input_magnitude`)."""


def single_feature_probe(n_features: int) -> Float[Array, "n_features n_features"]:
    """The single-feature probe: `eye(n_features) * 0.75`, one active feature per row."""
    return jnp.eye(n_features) * SINGLE_FEATURE_PROBE_MAGNITUDE


def single_feature_ci(
    lm: DecomposedModel,
    target: ResidMLPTarget,
    ci_fn: "CIFnCallable",
    n_features: int,
) -> dict[str, Array]:
    """Feed the single-feature probe (embedded through `W_E`) and read the `lower_leaky`
    CI per site, `{site: [n_features, C]}`."""
    resid = single_feature_probe(n_features) @ target.W_E
    return ci_fn(lm.site_inputs(target, resid)).lower
