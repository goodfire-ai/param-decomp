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
`(V, U)` (every site gets its own independent components).

`W_E` is the PREFIX: the residual entering the decomposed part is `x @ W_E` and
`resid_mlp_input_residual` maps `(W_E, x) -> x @ W_E`. The recon comparison is MSE on the
model OUTPUT `[B, n_features]` (NOT KL): `recon_loss_fn = resid_mlp_mse`.

Site weights are right-mult oriented like the LM targets (`site_out = x @ Wᵀ`): for
`mlp_in` `W` is `(d_mlp, d_embed)`; for `mlp_out` `(d_embed, d_mlp)`."""

from collections.abc import Callable
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

MLP_IN = "mlp_in"
MLP_OUT = "mlp_out"
KINDS = (MLP_IN, MLP_OUT)

EmbeddingMode = Literal["fixed_identity", "fixed_random", "learned"]
"""How `W_E`/`W_U` are obtained (torch had three regimes):
- `fixed_identity` — `W_E = W_U = I` (requires `n_features == d_embed`), the residual
  stream IS the feature basis (the unambiguous clean ground-truth regime).
- `fixed_random` — a fixed random unit-norm embedding, `W_U = W_Eᵀ`, frozen in pretrain.
- `learned` — `W_E`/`W_U` are trained alongside the MLP blocks (torch trainable embedding).
"""

LabelType = Literal["act_plus_resid", "abs"]
"""The pretrain read-off label (torch `label_type`): `act_fn(coeffs·x) + x` (the canonical
read-off) or `abs(coeffs·x)` (the |·| target)."""

LossType = Literal["readoff", "resid"]
"""What the pretrain objective compares (torch `loss_type`): the model OUTPUT against the
read-off labels (`readoff`), or the pre-unembed RESIDUAL against the embedded labels
(`resid`, `labels @ W_E`)."""


class CIFnCallable(Protocol):
    """The CI-fn surface the ResidMLP target-CI probe needs: `__call__(taps) -> CI` plus the
    `input_names` the probe feeds (satisfied by `ci_fn.LayerwiseMLPCIFn` / `GlobalMLPCIFn`)."""

    input_names: tuple[str, ...]

    def __call__(self, taps: dict[str, Array], *, remat: bool) -> CI: ...


@dataclass(frozen=True)
class ResidMLPConfig:
    n_features: int
    d_embed: int
    d_mlp: int
    n_layers: int
    act_fn_name: str
    in_bias: bool
    out_bias: bool
    embedding_mode: EmbeddingMode = "fixed_random"
    fixed_identity_embedding: bool | None = None
    """Legacy bool kept for the existing `run.py` call site: when set it DERIVES
    `embedding_mode` (`True -> fixed_identity`, `False -> fixed_random`). Pass
    `embedding_mode` directly for the `learned` regime. Exactly one of the two is given."""

    def __post_init__(self) -> None:
        if self.fixed_identity_embedding is not None:
            assert self.embedding_mode == "fixed_random", (
                "pass either embedding_mode or the legacy fixed_identity_embedding, not both"
            )
            derived: EmbeddingMode = (
                "fixed_identity" if self.fixed_identity_embedding else "fixed_random"
            )
            object.__setattr__(self, "embedding_mode", derived)
        if self.embedding_mode == "fixed_identity":
            assert self.n_features == self.d_embed, (self.n_features, self.d_embed)


@dataclass(frozen=True)
class ResidMLPTargetConfig:
    """The lab ResidMLP target config carried on `ExperimentConfig.target` (satisfies the
    core `TargetSites` protocol via `sites`). Pretrained from scratch in-process (no weight
    artifact); a fixed embedding (`W_U = W_Eᵀ`) with trainable per-layer MLP blocks.
    `global_batch` is the PD-step batch (the toy has no parquet `DataConfig`)."""

    n_features: int
    d_embed: int
    d_mlp: int
    n_layers: int
    act_fn_name: str
    in_bias: bool
    out_bias: bool
    fixed_identity_embedding: bool
    sites: tuple[SiteC, ...]
    pretrain_steps: int
    pretrain_batch_size: int
    pretrain_lr: float
    pretrain_seed: int
    feature_probability: float
    data_generation_type: str
    global_batch: int


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
    return clean_residual(target, resid) @ target.W_U


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
    """One site's output: the decomposed `site_out` if live, else the frozen `x @ Wᵀ`."""
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


class ResidMLPDecomposedModel(eqx.Module):
    """The ResidualMLP `DecomposedModel` (the `lm.py` contract; SPEC §1), positionless
    (`leading_axes=()`).

    Carries the FROZEN `ResidMLPTarget` weights as a field — threaded into the jitted step
    as a pytree arg, weights traced not baked. The TRAINABLE V/U (`vu: DecompVU`) is an
    explicit method arg, NOT a field (separate lifecycle). `sites` / `leading_axes` are
    static."""

    target: ResidMLPTarget
    sites: tuple[SiteSpec, ...] = eqx.field(static=True)
    leading_axes: tuple[str, ...] = eqx.field(static=True)

    @property
    def site_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.sites)

    def shardings(self, mesh: Mesh) -> "ResidMLPDecomposedModel":
        """Replicate every frozen leaf on the `dp` mesh — ResidualMLP weights are tiny."""
        repl = NamedSharding(mesh, P())
        return jax.tree.map(lambda _a: repl, self)

    @staticmethod
    def recon_loss_fn(
        masked_output: Float[Array, "B n_features"], clean_output: Float[Array, "B n_features"]
    ) -> Array:
        return resid_mlp_mse(masked_output, clean_output)

    def clean_output(self, resid: Float[Array, "B d_embed"]) -> Array:
        return clean_output(self.target, resid)

    def read_activations(
        self, resid: Float[Array, "B d_embed"], wanted: tuple[str, ...]
    ) -> dict[str, Array]:
        inputs = site_inputs(self.target, resid)
        return {k: inputs[k] for k in wanted}

    def prepare_compute_weights(self, vu: DecompVU) -> DecompVU:
        """Identity: ResidMLP weights are tiny + replicated, nothing to stack/gather/share."""
        return vu

    def masked_output(
        self,
        prepared: DecompVU,
        resid: Float[Array, "B d_embed"],
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
        return forward(prepared, resid, masks, delta_masks, routes)

    def masked_site_outputs(
        self,
        prepared: DecompVU,
        resid: Float[Array, "B d_embed"],
        masks: dict[str, Array],
        delta_masks: dict[str, Array],
        routes: dict[str, Array] | None,
        live: tuple[str, ...],
        has_delta: bool,
    ) -> dict[str, Array]:
        return masked_site_outputs(
            self.target, prepared, resid, masks, delta_masks, routes, live, has_delta
        )

    def weight_deltas(self, vu: DecompVU) -> dict[str, Array]:
        return weight_deltas_fp32(self.target, vu, self.sites)


def resid_mlp_decomposed_model(
    cfg: ResidMLPConfig, target: ResidMLPTarget, sites: tuple[SiteSpec, ...]
) -> ResidMLPDecomposedModel:
    """Wrap a pretrained `ResidMLPTarget` + decomposition config into the `DecomposedModel`."""
    sites = site_specs(cfg, tuple(SiteC(s.name, s.C) for s in sites))
    return ResidMLPDecomposedModel(target=target, sites=sites, leading_axes=())


def replicate_target[T: (ResidMLPTarget, ResidMLPDecomposedModel)](target: T, mesh: Mesh) -> T:
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
    """The pretrain-trainable subset: the per-layer MLP blocks, plus the `(W_E, W_U)`
    embedding pair when `embedding_mode == "learned"` (`None` for the fixed regimes, where
    the embedding is held constant). The pair is jointly present-or-absent (the embedding
    is never half-trained)."""

    layers: tuple[ResidMLPLayer, ...]
    embedding: tuple[Float[Array, "n_features d_embed"], Float[Array, "d_embed n_features"]] | None


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


def _init_embedding(
    cfg: ResidMLPConfig, key: Array
) -> tuple[Float[Array, "n_features d_embed"], Float[Array, "d_embed n_features"]]:
    """`(W_E, W_U)` per `embedding_mode`. `fixed_identity` -> `(I, I)`; `fixed_random` and
    `learned` start from a random unit-norm `W_E` with `W_U = W_Eᵀ` (`learned` then trains
    both independently from that init, torch's trainable-embedding regime)."""
    match cfg.embedding_mode:
        case "fixed_identity":
            assert cfg.n_features == cfg.d_embed, (cfg.n_features, cfg.d_embed)
            eye = jnp.eye(cfg.n_features)
            return eye, eye
        case "fixed_random" | "learned":
            raw = jax.random.normal(key, (cfg.n_features, cfg.d_embed))
            W_E = raw / jnp.linalg.norm(raw, axis=-1, keepdims=True)
            return W_E, W_E.T


def init_resid_mlp_target(cfg: ResidMLPConfig, key: Array) -> ResidMLPTarget:
    """Untrained ResidMLP target: the `embedding_mode` `W_E`/`W_U` and Kaiming-uniform MLP
    blocks. `fixed_identity` gives `W_E = W_U = I` (the residual stream IS the feature basis
    — the unambiguous clean ground-truth regime); `fixed_random`/`learned` start from a
    random unit-norm embedding (`W_U = W_Eᵀ`)."""
    embed_key, layers_key = jax.random.split(key)
    W_E, W_U = _init_embedding(cfg, embed_key)
    layer_keys = jax.random.split(layers_key, cfg.n_layers)
    layers = tuple(_init_layer(cfg, layer_keys[i]) for i in range(cfg.n_layers))
    return ResidMLPTarget(W_E=W_E, W_U=W_U, layers=layers, act_fn_name=cfg.act_fn_name)


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


def label_coeffs(n_features: int, use_trivial: bool, key: Array) -> Float[Array, " n_features"]:
    """Per-feature read-off coefficients (torch `calc_label_coeffs`): all-ones when
    `use_trivial`, else `U[1, 2)` (`rand(n_features) + 1`)."""
    if use_trivial:
        return jnp.ones((n_features,))
    return jax.random.uniform(key, (n_features,)) + 1.0


def feature_importances(n_features: int, importance_val: float) -> Float[Array, " n_features"]:
    """Geometric per-feature weighting (torch `compute_feature_importances`): feature `i`
    gets `importance_val ** i`. `importance_val == 1.0` is uniform (all ones)."""
    return importance_val ** jnp.arange(n_features, dtype=jnp.float32)


def readoff_labels(
    target: ResidMLPTarget, x: Float[Array, "B n_features"], coeffs: Float[Array, " n_features"]
) -> Float[Array, "B n_features"]:
    """The read-off pretrain target `act_fn(coeffs·x) + x` (torch
    `calc_act_plus_resid_labels`)."""
    return _act_fn(target.act_fn_name)(coeffs * x) + x


def abs_labels(
    x: Float[Array, "B n_features"], coeffs: Float[Array, " n_features"]
) -> Float[Array, "B n_features"]:
    """The `|coeffs·x|` pretrain target (torch `calc_abs_labels`)."""
    return jnp.abs(coeffs * x)


def pretrain_labels(
    target: ResidMLPTarget,
    x: Float[Array, "B n_features"],
    coeffs: Float[Array, " n_features"],
    label_type: LabelType,
) -> Float[Array, "B n_features"]:
    match label_type:
        case "act_plus_resid":
            return readoff_labels(target, x, coeffs)
        case "abs":
            return abs_labels(x, coeffs)


def clean_residual(target: ResidMLPTarget, resid: Float[Array, "B d_embed"]) -> Array:
    """The all-frozen forward up to (not including) the unembed — the pre-`W_U` residual
    `[B, d_embed]` (torch `ResidMLP.forward(return_residual=True)`). `clean_output` is this
    `@ W_U`."""
    act = _act_fn(target.act_fn_name)
    for block in target.layers:
        h = resid @ block.W_in.T
        if block.b_in is not None:
            h = h + block.b_in
        out = act(h) @ block.W_out.T
        if block.b_out is not None:
            out = out + block.b_out
        resid = resid + out
    return resid


def _trainable_target(trainable: "_ResidMLPTrainable", target: ResidMLPTarget) -> ResidMLPTarget:
    """Fold the trainable subset back into the full target: the MLP blocks always, and the
    `(W_E, W_U)` pair when the embedding is learned."""
    folded = eqx.tree_at(lambda t: t.layers, target, trainable.layers)
    if trainable.embedding is None:
        return folded
    W_E, W_U = trainable.embedding
    return eqx.tree_at(lambda t: (t.W_E, t.W_U), folded, (W_E, W_U))


def pretrain_resid_mlp_target(
    cfg: ResidMLPConfig,
    feature_probability: float,
    generation_type: str,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    label_type: LabelType = "act_plus_resid",
    loss_type: LossType = "readoff",
    use_trivial_label_coeffs: bool = True,
    importance_val: float = 1.0,
) -> ResidMLPTarget:
    """From-scratch pretrain of the frozen ResidMLP target (the read-off MSE objective
    `mean(((pred − labels)²) · feature_importances)`). The MLP blocks always train; the
    embedding `W_E`/`W_U` also trains when `embedding_mode == "learned"`, else it is held
    constant.

    - `label_type` picks the target (`act_plus_resid` read-off or `abs`).
    - `loss_type` picks `pred`: the model OUTPUT (`readoff`, compared to `labels`) or the
      pre-unembed RESIDUAL (`resid`, compared to the embedded labels `labels @ W_E`).
    - `use_trivial_label_coeffs` ones-coeffs vs `U[1, 2)`.
    - `importance_val` geometrically down-weights feature `i` by `importance_val ** i`.

    Deterministic in `seed`: this replaces the missing wandb pretrain checkpoint so a
    ResidMLP PD run is reproducible from the config alone."""
    import optax

    assert loss_type == "readoff" or importance_val == 1.0, (
        "feature_importances apply in feature space; the resid loss compares in d_embed space"
    )
    key = jax.random.PRNGKey(seed)
    init_key, coeff_key, data_key = jax.random.split(key, 3)
    target = init_resid_mlp_target(cfg, init_key)
    learned_embedding = (target.W_E, target.W_U) if cfg.embedding_mode == "learned" else None
    trainable = _ResidMLPTrainable(layers=target.layers, embedding=learned_embedding)
    coeffs = label_coeffs(cfg.n_features, use_trivial_label_coeffs, coeff_key)
    importances = feature_importances(cfg.n_features, importance_val)
    optimizer = optax.adamw(lr, weight_decay=0.0)
    opt_state = optimizer.init(eqx.filter(trainable, eqx.is_array))

    @jax.jit
    def step(
        trainable: "_ResidMLPTrainable", opt_state: optax.OptState, x: Array
    ) -> tuple["_ResidMLPTrainable", optax.OptState]:
        def loss_fn(trainable: "_ResidMLPTrainable") -> Array:
            frozen = _trainable_target(trainable, target)
            resid = x @ frozen.W_E
            labels = pretrain_labels(frozen, x, coeffs, label_type)
            match loss_type:
                case "readoff":
                    # feature-space comparison: geometric per-feature importances apply.
                    return jnp.mean(((clean_output(frozen, resid) - labels) ** 2) * importances)
                case "resid":
                    # residual-space (`d_embed`) comparison: feature importances do not map.
                    return jnp.mean((clean_residual(frozen, resid) - labels @ frozen.W_E) ** 2)

        grad = eqx.filter_grad(loss_fn)(trainable)
        updates, opt_state = optimizer.update(grad, opt_state, eqx.filter(trainable, eqx.is_array))
        return eqx.apply_updates(trainable, updates), opt_state

    for s in range(steps):
        x = sample_sparse_features(
            jax.random.fold_in(data_key, s), batch_size, cfg.n_features,
            feature_probability, generation_type,
        )  # fmt: skip
        trainable, opt_state = step(trainable, opt_state, x)
    return _trainable_target(trainable, target)


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
    lm: ResidMLPDecomposedModel,
    ci_fn: "CIFnCallable",
    n_features: int,
) -> dict[str, Array]:
    """Feed the single-feature probe (embedded through `W_E`) and read the `lower_leaky`
    CI per site, `{site: [n_features, C]}`."""
    resid = single_feature_probe(n_features) @ lm.target.W_E
    return ci_fn(lm.read_activations(resid, ci_fn.input_names), remat=False).lower
