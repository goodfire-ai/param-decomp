"""Vendored JAX TMS (Toy Model of Superposition) target — the first non-LM
`DecomposedModel`, positionless (`has_position_axis=False`; the waist is `[B, n_features]`).

Torch reference (read-only ground truth): `param_decomp/experiments/tms/models.py`.
The target is `out = relu(linear2(hidden_layers(linear1(x))))`: `linear1`
`(n_features -> n_hidden)` no bias, optional FROZEN `hidden_layers.{i}` `(n_hidden ->
n_hidden)` no bias, `linear2` `(n_hidden -> n_features)` with bias, with `linear1`/`linear2`
weights TIED (`linear2.weight = linear1.weight.T`). The DECOMPOSITION is UNTIED — each site
gets its own independent `(V, U)`.

The whole model is decomposed: the batch entering the decomposed model is the raw input
`x` `[B, n_features]`. The recon comparison is MSE on the post-ReLU `[B, n_features]`
output (NOT KL): `recon_loss_fn = tms_mse` (torch `recon_loss_mse`, mean over
batch × features).

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

from param_decomp.core.ci_fn import CI
from param_decomp.core.components import (
    ComponentStacks,
    SiteC,
    SiteDims,
    SiteSpec,
    site_slots_for,
)
from param_decomp.core.decomposed_linear import site_out
from param_decomp.core.masking import materialize_masking
from param_decomp.core.model import (
    EMPTY_CAPTURE_KEYS,
    CaptureKeys,
    ForwardResult,
    Masking,
)
from param_decomp.core.nonlinearity import Neurons
from param_decomp.core.placement import CIFnPlacement, PlacementRules
from param_decomp.targets.linear_site_capture import site_output_key

LINEAR1 = "linear1"
LINEAR2 = "linear2"

HiddenLayerInit = Literal["identity", "random"]

TMSGenerationType = Literal[
    "exactly_one_active",
    "exactly_two_active",
    "exactly_three_active",
    "exactly_four_active",
    "exactly_five_active",
    "at_least_zero_active",
]

_EXACTLY_N_ACTIVE: dict[str, int] = {
    "exactly_one_active": 1,
    "exactly_two_active": 2,
    "exactly_three_active": 3,
    "exactly_four_active": 4,
    "exactly_five_active": 5,
}


def hidden_layer_name(i: int) -> str:
    return f"hidden_layers.{i}"


def site_names_for(n_hidden_layers: int) -> tuple[str, ...]:
    """Canonical computation order: `linear1`, the `hidden_layers.{i}` in index order,
    then `linear2`."""
    hidden = tuple(hidden_layer_name(i) for i in range(n_hidden_layers))
    return (LINEAR1, *hidden, LINEAR2)


TMSCaptureSources = tuple[int, ...]


def site_input_tap_keys(sites: tuple[str, ...]) -> tuple[str, ...]:
    """Return the input taps for the sites in computation order."""
    assert sites, "TMS requires at least one site"
    return (sites[0], *(site_output_key(site) for site in sites[:-1]))


def _resolve_capture(sites: tuple[str, ...], keys: tuple[str, ...]) -> TMSCaptureSources:
    assert sites, "TMS requires at least one site"
    source_index = {sites[0]: 0} | {
        site_output_key(site): index + 1 for index, site in enumerate(sites)
    }
    try:
        sources = tuple(source_index[key] for key in keys)
    except KeyError as error:
        raise AssertionError(f"unknown TMS activation {error.args[0]!r}") from error
    assert len(set(sources)) == len(sources), (
        "multiple capture keys name one physical activation",
        keys,
        sources,
    )
    return sources


def _capture_key_by_index(keys: tuple[str, ...], sources: TMSCaptureSources) -> dict[int, str]:
    return dict(zip(sources, keys, strict=True))


def _record_capture(
    captures: dict[str, Array], requested: dict[int, str], index: int, value: Array
) -> None:
    key = requested.get(index)
    if key is not None:
        captures[key] = value


class CIFnCallable(Protocol):
    """The CI-fn surface the TMS target-CI probe needs: `__call__(taps) -> CI` plus the
    `capture_keys` the probe feeds (satisfied by `ci_fn.LayerwiseMLPCIFn` / `GlobalMLPCIFn`)."""

    @property
    def capture_keys(self) -> CaptureKeys: ...

    def __call__(
        self, taps: dict[str, Array], *, remat: bool, placement: CIFnPlacement | None
    ) -> CI: ...


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
    data_generation_type: TMSGenerationType
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


def site_dims(cfg: TMSConfig, name: str) -> SiteDims:
    """Matrix dimensions in right-mult orientation."""
    if name == LINEAR1:
        return SiteDims(d_in=cfg.n_features, d_out=cfg.n_hidden)
    if name == LINEAR2:
        return SiteDims(d_in=cfg.n_hidden, d_out=cfg.n_features)
    i = _parse_hidden_layer_index(name)
    assert i is not None and 0 <= i < cfg.n_hidden_layers, f"unknown TMS site {name!r}"
    return SiteDims(d_in=cfg.n_hidden, d_out=cfg.n_hidden)


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
        dims = site_dims(cfg, site.name)
        specs.append(
            SiteSpec(
                name=site.name,
                d_in=dims.d_in,
                d_out=dims.d_out,
                C=site.C,
                group=site.name,
                nonlinearity_partition=Neurons() if site.name == LINEAR2 else None,
            )
        )
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


def clean_forward(
    target: TMSTarget,
    resid: Float[Array, "B n_features"],
    capture_keys: tuple[str, ...],
    capture_sources: TMSCaptureSources,
) -> ForwardResult:
    if not capture_keys:
        return ForwardResult.from_producer(
            output=clean_output(target, resid), capture_keys=(), capture_values=()
        )
    requested = _capture_key_by_index(capture_keys, capture_sources)
    captures: dict[str, Array] = {}

    _record_capture(captures, requested, 0, resid)
    hidden = resid @ target.W1.T
    _record_capture(captures, requested, 1, hidden)
    for i, W in enumerate(target.hidden):
        hidden = hidden @ W.T
        _record_capture(captures, requested, i + 2, hidden)
    linear2 = hidden @ target.W2.T
    _record_capture(captures, requested, len(target.hidden) + 2, linear2)
    assert set(captures) == set(capture_keys), (sorted(captures), sorted(capture_keys))
    return ForwardResult.from_producer(
        output=jax.nn.relu(linear2 + target.b2),
        capture_keys=capture_keys,
        capture_values=tuple(captures[key] for key in capture_keys),
    )


def _run_masked(
    target: TMSTarget,
    components: ComponentStacks,
    resid: Float[Array, "B n_features"],
    component_masks: dict[str, Array],
    weight_delta_masks: dict[str, Array] | None,
    routes: dict[str, Array] | None,
    capture_keys: tuple[str, ...],
    capture_sources: TMSCaptureSources,
    placement: PlacementRules | None,
) -> ForwardResult:
    """Masked forward plus requested captures; every downstream site sees prior changes.

    Every site the forward visits is decomposed (`canonical_site_cs` pins the site set to
    `linear1`, every `hidden_layers.{i}`, `linear2`), so the masks must cover them exactly
    — there is no frozen-site path."""
    all_sites = set(site_names_for(len(target.hidden)))
    assert set(component_masks) == all_sites, (sorted(component_masks), sorted(all_sites))
    requested = _capture_key_by_index(capture_keys, capture_sources)
    captures: dict[str, Array] = {}
    _record_capture(captures, requested, 0, resid)

    def masked_site_output(site_name: str, W: Array, site_input: Array) -> Array:
        site_components = components.site(site_name)
        return site_out(
            site_input,
            site_components.V,
            site_components.U,
            W,
            component_masks[site_name],
            None if weight_delta_masks is None else weight_delta_masks[site_name],
            None if routes is None else routes[site_name],
            placement,
            None,
        )

    hidden = masked_site_output(LINEAR1, target.W1, resid)
    _record_capture(captures, requested, 1, hidden)
    for i, W in enumerate(target.hidden):
        hidden = masked_site_output(hidden_layer_name(i), W, hidden)
        _record_capture(captures, requested, i + 2, hidden)
    pre_relu = masked_site_output(LINEAR2, target.W2, hidden)
    _record_capture(captures, requested, len(target.hidden) + 2, pre_relu)
    pre_relu = pre_relu + target.b2
    assert set(captures) == set(capture_keys), (sorted(captures), sorted(capture_keys))
    return ForwardResult.from_producer(
        output=jax.nn.relu(pre_relu),
        capture_keys=capture_keys,
        capture_values=tuple(captures[key] for key in capture_keys),
    )


def weight_deltas_fp32(target: TMSTarget, components: ComponentStacks) -> dict[str, Array]:
    """fp32 `W − (V@U)ᵀ` per persistence stack, slot-aligned with `components.stacks`
    (SPEC N2; faithfulness input) — whole-stack einsum, never per-site `site()` slices."""
    out: dict[str, Array] = {}
    for shape, (Vs, Us) in components.stacks.items():
        Ws = jnp.stack(
            [
                _frozen_site_weight(target, name)
                for name, s, _slot in components.site_slots
                if s == shape
            ]
        )
        out[shape] = Ws.astype(jnp.float32) - jnp.einsum(
            "gic,gco->goi", Vs.astype(jnp.float32), Us.astype(jnp.float32)
        )
    return out


def tms_mse(
    masked_output: Float[Array, "B n_features"],
    clean_output: Float[Array, "B n_features"],
) -> Array:
    """MSE recon over the post-ReLU output, mean over batch × features (torch
    `recon_loss_mse`: `sum((pred-target)^2) / pred.numel()`), fp32."""
    masked_output = masked_output.astype(jnp.float32)
    clean_output = clean_output.astype(jnp.float32)
    return jnp.mean((masked_output - clean_output) ** 2)


class TMSDecomposedModel(eqx.Module):
    """The TMS `DecomposedModel` (the `model.py` contract; SPEC §1), positionless.

    Carries the FROZEN `TMSTarget` weights as a field — threaded into the jitted step as a
    pytree arg, weights traced not baked. The TRAINABLE V/U (`vu: ComponentStacks`) is an explicit
    method arg, NOT a field (separate lifecycle). `sites` / `has_position_axis` are static."""

    target: TMSTarget
    sites: tuple[SiteSpec, ...] = eqx.field(static=True)
    has_position_axis: bool = eqx.field(static=True)

    @property
    def site_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.sites)

    def shardings(self, placement: PlacementRules) -> "TMSDecomposedModel":
        """Replicate every frozen leaf on the `dp` mesh — TMS weights are tiny."""
        repl = NamedSharding(placement.mesh, P())
        return jax.tree.map(lambda _a: repl, self)

    @staticmethod
    def recon_loss_fn(
        masked_output: Float[Array, "B n_features"], clean_output: Float[Array, "B n_features"]
    ) -> Array:
        return tms_mse(masked_output, clean_output)

    def site_output_keys(self, sites: tuple[str, ...]) -> tuple[str, ...]:
        assert set(sites) <= set(self.site_names), (sites, self.site_names)
        return tuple(site_output_key(site) for site in sites)

    def assert_hidden_acts_reconstruction_points(self, keys: tuple[str, ...]) -> None:
        _resolve_capture(self.site_names, keys)

    def clean_forward(
        self,
        resid: Float[Array, "B n_features"],
        capture_keys: CaptureKeys = EMPTY_CAPTURE_KEYS,
        *,
        placement: PlacementRules | None,
    ) -> ForwardResult:
        del placement
        ordered_capture_keys = tuple(sorted(capture_keys))
        capture_sources = _resolve_capture(self.site_names, ordered_capture_keys)
        return clean_forward(self.target, resid, ordered_capture_keys, capture_sources)

    def prepare_compute_weights(
        self, vu: ComponentStacks, placement: PlacementRules | None
    ) -> ComponentStacks:
        """Identity: TMS weights are tiny + replicated, nothing to stack/gather/share."""
        del placement
        return vu

    def component_activation_forward(
        self,
        prepared_weights: ComponentStacks,
        inputs: Array,
        /,
        *,
        capture_keys: CaptureKeys,
        placement: PlacementRules | None,
    ) -> tuple[ForwardResult, dict[str, Array]]:
        del prepared_weights, inputs, capture_keys, placement
        raise NotImplementedError(
            f"{type(self).__name__} does not support component-activation harvest"
        )

    def stack_ci(self, ci_lower: dict[str, Array]) -> dict[str, Array]:
        return ci_lower

    def masked_forward(
        self,
        prepared_weights: ComponentStacks,
        resid: Float[Array, "B n_features"],
        /,
        *,
        masking: Masking,
        placement: PlacementRules | None,
        capture_keys: CaptureKeys = EMPTY_CAPTURE_KEYS,
        remat: bool,
    ) -> ForwardResult:
        ordered_capture_keys = tuple(sorted(capture_keys))
        capture_sources = _resolve_capture(self.site_names, ordered_capture_keys)
        explicit_masking = materialize_masking(masking)

        def forward(
            vu: ComponentStacks,
            resid: Array,
            component_masks: dict[str, Array],
            weight_delta_masks: dict[str, Array] | None,
            routes: dict[str, Array] | None,
        ) -> ForwardResult:
            return _run_masked(
                self.target,
                vu,
                resid,
                component_masks,
                weight_delta_masks,
                routes,
                ordered_capture_keys,
                capture_sources,
                placement,
            )

        forward = jax.checkpoint(forward) if remat else forward
        return forward(
            prepared_weights,
            resid,
            explicit_masking.component_masks,
            explicit_masking.weight_delta_masks,
            explicit_masking.routes,
        )

    def target_weight_sq_norms(self) -> dict[str, Array]:
        """Per-slot `‖W_s‖²` of each frozen stack, slot-aligned with `weight_deltas`
        (the S17 relative-error scales, read once at setup)."""
        norms: dict[str, list[Array]] = {}
        for name, group, _slot in site_slots_for(self.sites):
            frozen_weight = _frozen_site_weight(self.target, name)
            norms.setdefault(group, []).append(jnp.sum(frozen_weight.astype(jnp.float32) ** 2))
        return {group: jnp.stack(per_slot) for group, per_slot in norms.items()}

    def weight_deltas(self, vu: ComponentStacks) -> dict[str, Array]:
        return weight_deltas_fp32(self.target, vu)


def tms_decomposed_model(
    cfg: TMSConfig, target: TMSTarget, sites: tuple[SiteSpec, ...]
) -> TMSDecomposedModel:
    """Wrap a pretrained `TMSTarget` + decomposition config into the `DecomposedModel`."""
    sites = site_specs(cfg, tuple(SiteC(s.name, s.C) for s in sites))
    return TMSDecomposedModel(target=target, sites=sites, has_position_axis=False)


def replicate_target[T: (TMSTarget, TMSDecomposedModel)](target: T, mesh: Mesh) -> T:
    """Replicate the tiny frozen TMS weights on every device (the `replicate_frozen`
    analog; V/U / CI placement reuses the generic plan)."""
    repl = NamedSharding(mesh, P())
    return jax.tree.map(lambda a: jax.device_put(a, repl) if eqx.is_array(a) else a, target)


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


def sample_sparse_features(
    key: Array,
    batch: int,
    n_features: int,
    feature_probability: float,
    generation_type: TMSGenerationType,
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
    n = _EXACTLY_N_ACTIVE[generation_type]
    assert n <= n_features, f"cannot activate {n} of {n_features} features"
    sort_key = jax.random.uniform(gate_key, (batch, n_features))
    active = jnp.argsort(sort_key, axis=-1)[:, :n]  # n distinct features per row
    one_hot = jax.nn.one_hot(active, n_features).sum(axis=-2)  # [B, n_features], n ones per row
    return values * one_hot


def scatter_features(
    x_active: Float[Array, "B n_active"], active_indices: tuple[int, ...], n_features: int
) -> Float[Array, "B n_features"]:
    """Embed a batch sampled over ONLY the target features into full feature width — a
    targeted (tPD) run's TARGET stream (SPEC T2). The generator runs at `n_active` width
    so its generation type means what it says over the target features ("exactly one
    active" = one active TARGET feature; restricting a full-width sample after the fact
    would mostly produce empty rows). Non-target columns are identically zero."""
    assert active_indices == tuple(sorted(set(active_indices))), active_indices
    assert active_indices[0] >= 0 and active_indices[-1] < n_features, (
        f"active_indices {active_indices} out of range for n_features={n_features}"
    )
    assert x_active.shape[-1] == len(active_indices), (x_active.shape, active_indices)
    out = jnp.zeros((*x_active.shape[:-1], n_features), x_active.dtype)
    return out.at[..., jnp.array(active_indices)].set(x_active)


def pretrain_tms_target(
    cfg: TMSConfig,
    feature_probability: float,
    generation_type: TMSGenerationType,
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
            jax.random.fold_in(data_key, s),
            batch_size,
            cfg.n_features,
            feature_probability,
            generation_type,
        )
        trainable, opt_state = step(trainable, opt_state, x)
    return TMSTarget(W1=trainable.W1, hidden=hidden, W2=trainable.W1.T, b2=trainable.b2)


# ----------------------------- ground-truth target-CI eval -----------------------------


def identity_ci_error(ci_vals: Float[Array, "n_features C"], tolerance: float) -> int:
    """Discrete identity-CI distance (torch `IdentityCIPattern.distance_from`): permute
    columns toward identity (Hungarian on `-ci`), then over the FULL matrix minus the
    `min(shape)` block diagonal count entries `> tolerance` plus on-diagonal entries
    `< 1 - tolerance` (torch parity — trailing overcomplete columns/rows count as
    off-diagonal errors).

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

    off_diag_mask = np.ones(ci.shape, dtype=bool)
    off_diag_mask[:size, :size] &= ~np.eye(size, dtype=bool)
    off_diag_errors = int((ci[off_diag_mask] > tolerance).sum())
    on_diag_errors = int((np.diagonal(ci[:size, :size]) < (1 - tolerance)).sum())
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
    model: TMSDecomposedModel,
    ci_fn: "CIFnCallable",
    n_features: int,
) -> dict[str, Array]:
    """Feed the single-feature probe and read the `lower_leaky` CI per site,
    `{site: [n_features, C]}`."""
    probe = single_feature_probe(n_features)
    return ci_fn(
        model.clean_forward(probe, ci_fn.capture_keys, placement=None).captures,
        remat=False,
        placement=None,
    ).lower


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
