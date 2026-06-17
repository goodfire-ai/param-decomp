"""Forcing function for the generic model-I/O seam (issue #828).

The trainer's `[B,T,d]` residual is the fixed waist; only three EDGES are generic — the
model INPUT (`prefix_residual_fn` reads it), the model OUTPUT (`clean_output` /
`masked_output` return `Any`), and the recon comparison (`DecomposedModel.recon_loss_fn`).
This builds a tiny non-LM target that bends ALL THREE at once:

  * INPUT  — a dict `{"feat": [B,T,d], "gain": [B,T]}` rather than token ids.
  * OUTPUT — a tuple `(coords [B,T,k], aux [B,T,m])` rather than `[B,T,vocab]` logits.
  * LOSS   — a geometric MSE over the tuple rather than `kl_per_position`.

The site machinery is genuine: a real `[B,T,d]` residual, one decomposed site with V/U,
`[B,T,C]` masks, the frozen `x @ W` path for absent sites, real `weight_deltas`. It then
drives the actual `make_train_step` through a couple of steps and asserts the loss is
finite and the trainable state moves — locking the genericity against silent regression
to LM-only. The LM neutrality of these edges is proved separately by the stacked-parity /
equivalence goldens passing unchanged.
"""

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import random
from jaxtyping import Array, Float

from jax_single_pool.ci_fn import CIArch, init_ci_fn
from jax_single_pool.lm import DecomposedModel, SiteSpec
from jax_single_pool.recon import build_recon_terms
from jax_single_pool.train import TrainState, make_train_step
from param_decomp_config.losses import (
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    StochasticReconLossConfig,
)

B, T, D, C = 2, 3, 8, 5
K_COORDS, M_AUX = 4, 2
SITE = "block.0.proj"


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SyntheticFrozen:
    """The frozen target: one `[D,D]` site weight plus two readout heads."""

    W: Float[Array, "D D"]
    read_coords: Float[Array, "K D"]
    read_aux: Float[Array, "M D"]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SyntheticPrefix:
    feat_proj: Float[Array, "D D"]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SyntheticComponents:
    V: Float[Array, "D C"]
    U: Float[Array, "C D"]


def _heads(frozen: SyntheticFrozen, hidden: Array) -> tuple[Array, Array]:
    return hidden @ frozen.read_coords.T, hidden @ frozen.read_aux.T


def _synthetic_lm() -> DecomposedModel:
    site = SiteSpec(name=SITE, d_in=D, d_out=D, C=C)

    def clean_output(frozen: SyntheticFrozen, resid: Array) -> tuple[Array, Array]:
        return _heads(frozen, resid @ frozen.W.T)

    def site_inputs(_frozen: SyntheticFrozen, resid: Array) -> dict[str, Array]:
        return {SITE: resid}

    def masked_output(
        frozen: SyntheticFrozen,
        components: SyntheticComponents,
        resid: Array,
        masks: dict[str, Array],
        delta_masks: dict[str, Array],
        routes: dict[str, Array] | None,
        live: tuple[str, ...],
        has_delta: bool,
    ) -> tuple[Array, Array]:
        assert live == (SITE,) and routes is None, (live, routes)
        V, U, W = components.V, components.U, frozen.W
        hidden = (resid @ V) * masks[SITE] @ U
        if has_delta:
            delta = W - (V @ U).T
            hidden = hidden + delta_masks[SITE][..., None] * (resid @ delta.T)
        return _heads(frozen, hidden)

    def masked_site_outputs(
        frozen: SyntheticFrozen,
        components: SyntheticComponents,
        resid: Array,
        masks: dict[str, Array],
        delta_masks: dict[str, Array],
        routes: dict[str, Array] | None,
        live: tuple[str, ...],
        has_delta: bool,
    ) -> dict[str, Array]:
        assert live == (SITE,) and routes is None, (live, routes)
        V, U, W = components.V, components.U, frozen.W
        hidden = (resid @ V) * masks[SITE] @ U
        if has_delta:
            delta = W - (V @ U).T
            hidden = hidden + delta_masks[SITE][..., None] * (resid @ delta.T)
        return {SITE: hidden}

    def weight_deltas(frozen: SyntheticFrozen, components: SyntheticComponents) -> dict[str, Array]:
        return {
            SITE: frozen.W.astype(jnp.float32)
            - (components.V.astype(jnp.float32) @ components.U.astype(jnp.float32)).T
        }

    def geometric_mse(masked: tuple[Array, Array], clean: tuple[Array, Array]) -> Float[Array, ""]:
        """Non-KL recon: mean squared error over both tuple heads, fp32, per position."""
        coords_err = (masked[0].astype(jnp.float32) - clean[0].astype(jnp.float32)) ** 2
        aux_err = (masked[1].astype(jnp.float32) - clean[1].astype(jnp.float32)) ** 2
        return (jnp.sum(coords_err) + jnp.sum(aux_err)) / (B * T)

    return DecomposedModel(
        sites=(site,),
        leading_axes=("sequence",),
        clean_output=clean_output,
        site_inputs=site_inputs,
        masked_output=masked_output,
        masked_site_outputs=masked_site_outputs,
        weight_deltas=weight_deltas,
        recon_loss_fn=geometric_mse,
    )


def prefix_residual(prefix: SyntheticPrefix, inputs: dict[str, Array]) -> Array:
    """Input edge: a DICT batch -> the `[B,T,D]` residual (not token ids)."""
    return (inputs["feat"] @ prefix.feat_proj.T) * inputs["gain"][..., None]


def test_default_recon_loss_fn_is_kl_per_position():
    """An LM target that omits `recon_loss_fn` gets `kl_per_position` — the LM path is
    byte-for-byte the pre-seam behavior (proved at scale by the stacked-parity golden)."""
    from jax_single_pool.losses import kl_per_position

    lm = DecomposedModel(
        sites=(SiteSpec(SITE, D, D, C),),
        leading_axes=("sequence",),
        clean_output=lambda f, r: r,
        site_inputs=lambda f, r: {SITE: r},
        masked_output=lambda *a: a[2],
        masked_site_outputs=lambda *a: {SITE: a[2]},
        weight_deltas=lambda f, c: {},
    )
    assert lm.recon_loss_fn is kl_per_position


def test_prefix_residual_consumes_dict_input():
    """The input edge accepts the loader's native batch (here a dict), not just tokens."""
    key = random.PRNGKey(0)
    prefix = SyntheticPrefix(feat_proj=random.normal(key, (D, D)))
    inputs = {
        "feat": random.normal(random.fold_in(key, 1), (B, T, D)),
        "gain": random.uniform(random.fold_in(key, 2), (B, T)),
    }
    resid = jax.jit(prefix_residual)(prefix, inputs)
    assert resid.shape == (B, T, D)


def test_tuple_output_and_geometric_loss_flow():
    """`clean_output`/`masked_output` emit a tuple; `recon_loss_fn` (MSE) contracts it."""
    lm = _synthetic_lm()
    key = random.PRNGKey(1)
    frozen = SyntheticFrozen(
        W=random.normal(random.fold_in(key, 0), (D, D)),
        read_coords=random.normal(random.fold_in(key, 1), (K_COORDS, D)),
        read_aux=random.normal(random.fold_in(key, 2), (M_AUX, D)),
    )
    components = SyntheticComponents(
        V=random.normal(random.fold_in(key, 3), (D, C)) * 0.1,
        U=random.normal(random.fold_in(key, 4), (C, D)) * 0.1,
    )
    resid = random.normal(random.fold_in(key, 5), (B, T, D))

    clean = lm.clean_output(frozen, resid)
    assert isinstance(clean, tuple) and len(clean) == 2
    assert clean[0].shape == (B, T, K_COORDS) and clean[1].shape == (B, T, M_AUX)

    masks = {SITE: jnp.ones((B, T, C))}
    delta_masks = {SITE: jnp.zeros((B, T))}
    masked = lm.masked_output(frozen, components, resid, masks, delta_masks, None, (SITE,), False)
    assert isinstance(masked, tuple) and len(masked) == 2

    loss = lm.recon_loss_fn(masked, clean)
    assert loss.shape == () and jnp.isfinite(loss)


def _initial_state(lm: DecomposedModel, components: SyntheticComponents, ci_arch: CIArch):
    opt_vu = optax.adamw(1e-2, weight_decay=0.0)
    opt_ci = optax.adamw(1e-2, weight_decay=0.0)
    ci_fn = init_ci_fn(ci_arch, lm.sites, random.PRNGKey(11))
    state = TrainState(
        components=components,
        ci_fn=ci_fn,
        components_opt_state=opt_vu.init(eqx.filter(components, eqx.is_array)),
        ci_fn_opt_state=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
        sources={},
        sources_opt_state={},
        step=jnp.zeros((), jnp.int32),
    )
    return state, opt_vu, opt_ci


def test_train_step_runs_through_generic_target():
    """End-to-end: the real `make_train_step` drives the synthetic dict-in/tuple-out/MSE
    target for two steps; the loss stays finite and the trainable V/U actually move."""
    lm = _synthetic_lm()
    key = random.PRNGKey(2)
    frozen = SyntheticFrozen(
        W=random.normal(random.fold_in(key, 0), (D, D)),
        read_coords=random.normal(random.fold_in(key, 1), (K_COORDS, D)),
        read_aux=random.normal(random.fold_in(key, 2), (M_AUX, D)),
    )
    components = SyntheticComponents(
        V=random.normal(random.fold_in(key, 3), (D, C)) * 0.1,
        U=random.normal(random.fold_in(key, 4), (C, D)) * 0.1,
    )
    resid = random.normal(random.fold_in(key, 5), (B, T, D))

    ci_arch = CIArch(d_model=8, n_blocks=1, n_heads=2, mlp_hidden=16)
    state, opt_vu, opt_ci = _initial_state(lm, components, ci_arch)

    loss_spec = build_recon_terms(
        (
            FaithfulnessLossConfig(coeff=1.0),
            ImportanceMinimalityLossConfig(coeff=1e-4, pnorm=2.0, beta=0.0, p_anneal_final_p=1.0),
            StochasticReconLossConfig(coeff=1.0),
        ),
        lm.site_names,
        n_mask_samples=1,
        sampling="continuous",
    )
    step_fn = make_train_step(
        lm=lm,
        loss_spec=loss_spec,
        components_optimizer=opt_vu,
        ci_fn_optimizer=opt_ci,
        total_steps=10,
        remat_recon_forwards=False,
        mesh=None,
    )

    V_before = state.components.V
    run_key = random.PRNGKey(3)
    for step_idx in range(2):
        state, metrics = step_fn(state, frozen, resid, random.fold_in(run_key, step_idx))
        assert jnp.isfinite(metrics["total"]), (step_idx, metrics["total"])
        assert "loss/StochasticReconLoss" in metrics

    assert not jnp.allclose(state.components.V, V_before), "V did not move — step is a no-op"
