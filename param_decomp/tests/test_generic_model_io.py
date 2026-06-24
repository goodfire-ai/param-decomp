"""Forcing function for the generic model-I/O seam (issue #828).

The trainer's `[B,T,d]` residual is the fixed waist; only three EDGES are generic — the
model INPUT (`prefix_residual_fn` reads it), the model OUTPUT (`clean_output` /
`masked_output` return `Any`), and the recon comparison (`DecomposedModel.recon_loss_fn`).
The trainable components are NOT a generic edge: every target carries the universal
`DecompVU` V/U pytree, so this synthetic target uses it too. This builds a tiny non-LM
target that bends the three real edges at once:

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

from param_decomp.ci_fn import (
    Chunk,
    ChunkwiseTransformerCIArch,
    build_ci_fn,
)
from param_decomp.components import DecompVU, SiteSpec
from param_decomp.configs import (
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    StochasticReconLossConfig,
)
from param_decomp.lm import DecomposedModel
from param_decomp.recon import build_loss_terms
from param_decomp.train import TrainState, make_train_step

B, T, D, C = 2, 3, 8, 5
K_COORDS, M_AUX = 4, 2
SITE = "block.0.proj"


class SyntheticDecomposedModel(eqx.Module):
    """A non-LM `DecomposedModel`: dict input, tuple `(coords, aux)` output, geometric-MSE
    recon. Carries its frozen target weights (`W` + two readout heads) as array fields; the
    trainable V/U (the universal `DecompVU`) stays an explicit method arg."""

    W: Float[Array, "D D"]
    read_coords: Float[Array, "K D"]
    read_aux: Float[Array, "M D"]
    sites: tuple[SiteSpec, ...] = eqx.field(static=True)
    leading_axes: tuple[str, ...] = eqx.field(static=True)

    @property
    def site_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.sites)

    def shardings(self, mesh: "jax.sharding.Mesh") -> "SyntheticDecomposedModel":
        repl = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        return jax.tree.map(lambda _a: repl, self)

    @staticmethod
    def recon_loss_fn(
        masked_output: tuple[Array, Array], clean_output: tuple[Array, Array]
    ) -> Float[Array, ""]:
        """Non-KL recon: mean squared error over both tuple heads, fp32, per position."""
        coords_err = (
            masked_output[0].astype(jnp.float32) - clean_output[0].astype(jnp.float32)
        ) ** 2
        aux_err = (masked_output[1].astype(jnp.float32) - clean_output[1].astype(jnp.float32)) ** 2
        return (jnp.sum(coords_err) + jnp.sum(aux_err)) / (B * T)

    def _heads(self, hidden: Array) -> tuple[Array, Array]:
        return hidden @ self.read_coords.T, hidden @ self.read_aux.T

    def clean_output(self, resid: Array) -> tuple[Array, Array]:
        return self._heads(resid @ self.W.T)

    def read_activations(self, resid: Array, wanted: tuple[str, ...]) -> dict[str, Array]:
        assert wanted == (SITE,), wanted
        return {SITE: resid}

    def masked_output(
        self,
        vu: DecompVU,
        resid: Array,
        masks: dict[str, Array],
        delta_masks: dict[str, Array],
        routes: dict[str, Array] | None,
        live: tuple[str, ...],
        has_delta: bool,
    ) -> tuple[Array, Array]:
        assert live == (SITE,) and routes is None, (live, routes)
        V, U = vu.site(SITE)
        W = self.W
        hidden = (resid @ V) * masks[SITE] @ U
        if has_delta:
            delta = W - (V @ U).T
            hidden = hidden + delta_masks[SITE][..., None] * (resid @ delta.T)
        return self._heads(hidden)

    def masked_site_outputs(
        self,
        vu: DecompVU,
        resid: Array,
        masks: dict[str, Array],
        delta_masks: dict[str, Array],
        routes: dict[str, Array] | None,
        live: tuple[str, ...],
        has_delta: bool,
    ) -> dict[str, Array]:
        assert live == (SITE,) and routes is None, (live, routes)
        V, U = vu.site(SITE)
        W = self.W
        hidden = (resid @ V) * masks[SITE] @ U
        if has_delta:
            delta = W - (V @ U).T
            hidden = hidden + delta_masks[SITE][..., None] * (resid @ delta.T)
        return {SITE: hidden}

    def weight_deltas(self, vu: DecompVU) -> dict[str, Array]:
        V, U = vu.site(SITE)
        return {
            SITE: self.W.astype(jnp.float32) - (V.astype(jnp.float32) @ U.astype(jnp.float32)).T
        }


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SyntheticPrefix:
    feat_proj: Float[Array, "D D"]


def _synthetic_lm(key: jax.Array) -> SyntheticDecomposedModel:
    return SyntheticDecomposedModel(
        W=random.normal(random.fold_in(key, 0), (D, D)),
        read_coords=random.normal(random.fold_in(key, 1), (K_COORDS, D)),
        read_aux=random.normal(random.fold_in(key, 2), (M_AUX, D)),
        sites=(SiteSpec(name=SITE, d_in=D, d_out=D, C=C),),
        leading_axes=("sequence",),
    )


def _synthetic_vu(key: jax.Array) -> DecompVU:
    V = random.normal(random.fold_in(key, 3), (D, C)) * 0.1
    U = random.normal(random.fold_in(key, 4), (C, D)) * 0.1
    return DecompVU(vu={SITE: (V, U)})


def prefix_residual(prefix: SyntheticPrefix, inputs: dict[str, Array]) -> Array:
    """Input edge: a DICT batch -> the `[B,T,D]` residual (not token ids)."""
    return (inputs["feat"] @ prefix.feat_proj.T) * inputs["gain"][..., None]


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
    key = random.PRNGKey(1)
    lm = _synthetic_lm(key)
    components = _synthetic_vu(key)
    resid = random.normal(random.fold_in(key, 5), (B, T, D))

    clean = lm.clean_output(resid)
    assert isinstance(clean, tuple) and len(clean) == 2
    assert clean[0].shape == (B, T, K_COORDS) and clean[1].shape == (B, T, M_AUX)

    masks = {SITE: jnp.ones((B, T, C))}
    delta_masks = {SITE: jnp.zeros((B, T))}
    masked = lm.masked_output(components, resid, masks, delta_masks, None, (SITE,), False)
    assert isinstance(masked, tuple) and len(masked) == 2

    loss = lm.recon_loss_fn(masked, clean)
    assert loss.shape == () and jnp.isfinite(loss)


def _initial_state(lm: DecomposedModel, components: DecompVU, ci_arch: ChunkwiseTransformerCIArch):
    opt_vu = optax.adamw(1e-2, weight_decay=0.0)
    opt_ci = optax.adamw(1e-2, weight_decay=0.0)
    ci_fn = build_ci_fn(ci_arch, lm.sites, random.PRNGKey(11))
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
    key = random.PRNGKey(2)
    lm = _synthetic_lm(key)
    components = _synthetic_vu(key)
    resid = random.normal(random.fold_in(key, 5), (B, T, D))

    ci_arch = ChunkwiseTransformerCIArch(
        chunks=(Chunk(input_taps=(SITE,), output_sites=(SITE,)),),
        input_dim=D,
        d_model=8,
        n_blocks=1,
        n_heads=2,
        mlp_hidden=16,
    )
    state, opt_vu, opt_ci = _initial_state(lm, components, ci_arch)

    loss_terms = build_loss_terms(
        (
            FaithfulnessLossConfig(coeff=1.0),
            ImportanceMinimalityLossConfig(coeff=1e-4, pnorm=2.0, beta=0.0, p_anneal_final_p=1.0),
            StochasticReconLossConfig(coeff=1.0),
        ),
        lm.site_names,
    )
    step_fn = make_train_step(
        lm=lm,
        loss_terms=loss_terms,
        components_optimizer=opt_vu,
        ci_fn_optimizer=opt_ci,
        total_steps=10,
        remat_recon_forwards=False,
        mesh=None,
    )

    V_before = state.components.site(SITE)[0]
    run_key = random.PRNGKey(3)
    for step_idx in range(2):
        state, metrics = step_fn(lm, state, resid, random.fold_in(run_key, step_idx))
        assert jnp.isfinite(metrics["total"]), (step_idx, metrics["total"])
        assert "loss/StochasticReconLoss" in metrics

    assert not jnp.allclose(state.components.site(SITE)[0], V_before), (
        "V did not move — step is a no-op"
    )
