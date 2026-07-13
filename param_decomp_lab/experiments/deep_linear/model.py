"""Deep-linear identity target — one-hot inputs through `n_layers` FROZEN identity
layers to a softmax readout; positionless (`leading_axes=()`, the waist is
`[B, n_features]`).

The target is `logits = x @ I.T @ ... @ I.T` — every site `layers.{i}` is hardcoded to
`eye(n_features)`, nothing is pretrained. The recon comparison is the LM's
`kl_per_position` on the final logits (the softmax lives inside the KL): with one-hot
inputs the clean logit vector IS the input, so the decomposition's ground truth at every
site is the identity — C live rank-1 components `e_i e_iᵀ`, one per feature, with
one-hot per-feature CI (the TMS `identity_ci_error` pattern).
"""

from dataclasses import dataclass
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float

from param_decomp.components import DecompVU, SiteC, SiteSpec, site_out
from param_decomp.lm import run_stochastic_masked_output
from param_decomp.losses import kl_per_position

LAYER_PREFIX = "layers."


def layer_name(i: int) -> str:
    return f"{LAYER_PREFIX}{i}"


def site_names_for(n_layers: int) -> tuple[str, ...]:
    """Canonical computation order: `layers.0` … `layers.{n_layers-1}`."""
    return tuple(layer_name(i) for i in range(n_layers))


def _parse_layer_index(name: str) -> int:
    assert name.startswith(LAYER_PREFIX), f"unknown deep-linear site {name!r}"
    return int(name[len(LAYER_PREFIX) :])


ReconKind = Literal["kl", "mse"]


@dataclass(frozen=True)
class DeepLinearConfig:
    """`logit_scale` multiplies the FINAL output only (a softmax temperature for the KL
    recon — the layers stay exact identities, so the per-site ground truth is unchanged);
    `recon` picks the recon comparison (`kl` = LM semantics, `mse` = raw logits)."""

    n_features: int
    n_layers: int
    logit_scale: float = 1.0
    recon: ReconKind = "kl"


@dataclass(frozen=True)
class DeepLinearTargetConfig:
    """The lab deep-linear target config carried on `ExperimentConfig.target` (satisfies
    the core `TargetSites` protocol via `sites`). The target is CONSTRUCTED (identity
    weights), not pretrained — a run is reproducible from the config alone. `global_batch`
    is the PD-step batch (no parquet `DataConfig`; the toy samples one-hot rows on the
    fly)."""

    n_features: int
    n_layers: int
    logit_scale: float
    recon: ReconKind
    sites: tuple[SiteC, ...]
    global_batch: int


class DeepLinearTarget(eqx.Module):
    """Frozen identity weights, right-mult oriented like every other target
    (`site_out = x @ W.T`; the orientation is moot for `eye`)."""

    layers: tuple[Float[Array, "n_features n_features"], ...]


def init_deep_linear_target(cfg: DeepLinearConfig) -> DeepLinearTarget:
    return DeepLinearTarget(layers=tuple(jnp.eye(cfg.n_features) for _ in range(cfg.n_layers)))


def canonical_site_cs(site_cs: tuple[SiteC, ...]) -> tuple[SiteC, ...]:
    """Canonical order is `layers.{i}` in index order (= computation order). The layer
    count is inferred from the configured names; each must appear exactly once."""
    by_name = {s.name: s for s in site_cs}
    assert len(by_name) == len(site_cs), f"duplicate site in {[s.name for s in site_cs]}"
    expected = site_names_for(len(site_cs))
    assert sorted(by_name) == sorted(expected), (
        f"deep-linear sites must be {expected}, got {sorted(by_name)}"
    )
    return tuple(by_name[name] for name in expected)


def expand_wildcard_site_cs(entries: tuple[SiteC, ...], n_layers: int) -> tuple[SiteC, ...]:
    """A single `layers.*` entry expands to every layer at that entry's C (the
    `module_pattern` wildcard convention — a 200-layer stack is one config stanza);
    explicit `layers.{i}` entries pass through. Result is canonical-ordered."""
    expanded: list[SiteC] = []
    for entry in entries:
        if entry.name == f"{LAYER_PREFIX}*":
            expanded.extend(SiteC(layer_name(i), entry.C) for i in range(n_layers))
        else:
            _parse_layer_index(entry.name)
            expanded.append(entry)
    assert len(expanded) == n_layers, f"{len(expanded)} expanded sites but n_layers={n_layers}"
    return canonical_site_cs(tuple(expanded))


def site_specs(cfg: DeepLinearConfig, site_cs: tuple[SiteC, ...]) -> tuple[SiteSpec, ...]:
    site_cs = canonical_site_cs(site_cs)
    assert len(site_cs) == cfg.n_layers, f"config n_layers={cfg.n_layers} but {len(site_cs)} sites"
    return tuple(SiteSpec(s.name, cfg.n_features, cfg.n_features, s.C) for s in site_cs)


def clean_output(target: DeepLinearTarget, resid: Float[Array, "B n_features"]) -> Array:
    """The all-frozen forward — the final LOGITS (the softmax lives in `kl_per_position`)."""
    for W in target.layers:
        resid = resid @ W.T
    return resid


def site_inputs(target: DeepLinearTarget, resid: Float[Array, "B n_features"]) -> dict[str, Array]:
    """Clean CI inputs per site (SPEC S4): each `layers.{i}` reads the frozen output of
    the chain up to it."""
    inputs: dict[str, Array] = {}
    for i, W in enumerate(target.layers):
        inputs[layer_name(i)] = resid
        resid = resid @ W.T
    return inputs


def _run_masked(
    target: DeepLinearTarget,
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
    forward; the rest run the frozen `x @ W.T` path. Each layer reads the (possibly
    masked) output of the layer before it, so a masked site propagates."""
    live_set = frozenset(live)
    for i, W in enumerate(target.layers):
        site = layer_name(i)
        if site not in live_set:
            resid = resid @ W.T
            continue
        V, U = components.site(site)
        resid = site_out(
            resid, V, U, W, masks[site], delta_masks[site] if has_delta else None,
            None if routes is None else routes[site],
        )  # fmt: skip
        if collect is not None:
            collect[site] = resid
    return resid


def weight_deltas_fp32(
    target: DeepLinearTarget, components: DecompVU, sites: tuple[SiteSpec, ...]
) -> dict[str, Array]:
    """fp32 `W − V@U` per site from fp32 masters (SPEC N2; faithfulness input)."""
    out: dict[str, Array] = {}
    for spec in sites:
        W = target.layers[_parse_layer_index(spec.name)]
        V, U = components.site(spec.name)
        out[spec.name] = W.astype(jnp.float32) - (V.astype(jnp.float32) @ U.astype(jnp.float32)).T
    return out


class DeepLinearDecomposedModel(eqx.Module):
    """The deep-linear `DecomposedModel` (the `lm.py` contract; SPEC §1), positionless
    (`leading_axes=()`).

    Carries the FROZEN `DeepLinearTarget` weights as a field — threaded into the jitted
    step as a pytree arg, weights traced not baked. The TRAINABLE V/U (`vu: DecompVU`) is
    an explicit method arg, NOT a field (separate lifecycle)."""

    target: DeepLinearTarget
    sites: tuple[SiteSpec, ...] = eqx.field(static=True)
    leading_axes: tuple[str, ...] = eqx.field(static=True)
    logit_scale: float = eqx.field(static=True)
    recon: ReconKind = eqx.field(static=True)

    @property
    def site_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.sites)

    def shardings(self, mesh: Mesh) -> "DeepLinearDecomposedModel":
        """Replicate every frozen leaf on the mesh — the identity weights are tiny."""
        repl = NamedSharding(mesh, P())
        return jax.tree.map(lambda _a: repl, self)

    def recon_loss_fn(
        self,
        masked_output: Float[Array, "B n_features"],
        clean_output: Float[Array, "B n_features"],
    ) -> Array:
        """Pure over static config only (no arrays closed over) — safe for the jitted step."""
        if self.recon == "kl":
            return kl_per_position(masked_output, clean_output)
        masked_output = masked_output.astype(jnp.float32)
        clean_output = clean_output.astype(jnp.float32)
        return jnp.mean((masked_output - clean_output) ** 2)

    def clean_output(self, resid: Float[Array, "B n_features"]) -> Array:
        return clean_output(self.target, resid) * self.logit_scale

    def read_activations(
        self, resid: Float[Array, "B n_features"], wanted: tuple[str, ...]
    ) -> dict[str, Array]:
        inputs = site_inputs(self.target, resid)
        return {k: inputs[k] for k in wanted}

    def prepare_compute_weights(self, vu: DecompVU) -> DecompVU:
        """Identity: the weights are tiny + replicated, nothing to stack/gather/share."""
        return vu

    def masked_output(
        self,
        prepared: DecompVU,
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
            return (
                _run_masked(
                    self.target, vu, resid, masks, delta_masks, routes, live, has_delta, None
                )
                * self.logit_scale
            )

        forward = jax.checkpoint(forward) if remat else forward
        return forward(prepared, resid, masks, delta_masks, routes)

    def stack_ci(self, ci_lower: dict[str, Array]) -> dict[str, Array]:
        return ci_lower

    def masked_output_stochastic(
        self,
        prepared: DecompVU,
        resid: Float[Array, "B n_features"],
        ci_stacked: dict[str, Array],
        draw_key: Array,
        routes: dict[str, Array] | None,
        live: tuple[str, ...],
        has_delta: bool,
        *,
        remat: bool,
    ) -> Array:
        return run_stochastic_masked_output(
            self, prepared, resid, ci_stacked, draw_key, routes, live, has_delta, remat=remat
        )

    def masked_site_outputs(
        self,
        prepared: DecompVU,
        resid: Float[Array, "B n_features"],
        masks: dict[str, Array],
        delta_masks: dict[str, Array],
        routes: dict[str, Array] | None,
        live: tuple[str, ...],
        has_delta: bool,
    ) -> dict[str, Array]:
        collect: dict[str, Array] = {}
        _run_masked(
            self.target, prepared, resid, masks, delta_masks, routes, live, has_delta, collect
        )
        assert set(collect) == set(live), (sorted(collect), sorted(live))
        return collect

    def weight_deltas(self, vu: DecompVU) -> dict[str, Array]:
        return weight_deltas_fp32(self.target, vu, self.sites)


def deep_linear_decomposed_model(
    cfg: DeepLinearConfig, target: DeepLinearTarget, sites: tuple[SiteC, ...]
) -> DeepLinearDecomposedModel:
    return DeepLinearDecomposedModel(
        target=target,
        sites=site_specs(cfg, sites),
        leading_axes=(),
        logit_scale=cfg.logit_scale,
        recon=cfg.recon,
    )


def replicate_target[T: (DeepLinearTarget, DeepLinearDecomposedModel)](target: T, mesh: Mesh) -> T:
    """Replicate the tiny frozen identity weights on every device."""
    repl = NamedSharding(mesh, P())
    return jax.tree.map(lambda a: jax.device_put(a, repl) if eqx.is_array(a) else a, target)


def sample_one_hot(key: Array, batch: int, n_features: int) -> Float[Array, "B n_features"]:
    """Uniform one-hot rows — the toy's whole data distribution."""
    return jax.nn.one_hot(jax.random.randint(key, (batch,), 0, n_features), n_features)


def one_hot_probe(n_features: int) -> Float[Array, "n_features n_features"]:
    """The per-feature CI probe: every one-hot input, magnitude 1.0 (the actual data
    values — unlike TMS's 0.75, deep-linear inputs are exactly one-hot)."""
    return jnp.eye(n_features)
