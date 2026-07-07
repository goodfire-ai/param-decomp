"""`DecomposedModel` — the interface a vendored target implements for the generic trainer.

The trainer (`train.py`) is abstract over the target model: it sees an ordered set of
decomposed **sites** (SPEC §1.2) and a handful of methods on the model `eqx.Module`. The
model carries its FROZEN target weights as fields; the TRAINABLE V/U (`vu`) is passed to
the forward methods explicitly (separate lifecycle). Everything at the boundary is keyed
by site name (flat dicts, torch-module-path style); how a target lays its parameters out
internally (e.g. the Llama target's stacked layer axis) is its own business.

The activation WAIST is generic: per decomposed site activations are `[*leading, d]`
and masks/CI are `[*leading, C]`, where `leading = (batch,) + named position axes`
(`leading_axes` names the position axes; `("sequence",)` for an LM, `()` for TMS). Batch
is ever-present and semantics-free (the data/shard axis); CI is always independent over
every leading axis. Masking, routing, source scopes, imp-min, and normalization all
operate over the opaque `*leading` prefix. The three EDGES are generic too — the model's
INPUT (whatever `read_activations`/`clean_output`/`masked_output` read upstream of the
residual; tokens for an LM, a dict for a bio target), the model's OUTPUT
(`clean_output`/`masked_output` return `Any` — logits, a tuple of heads, coords), and the
recon comparison (`recon_loss_fn`, `kl_per_position` for an LM).

The frozen weights ride on the model `eqx.Module` and reach the jitted step as a pytree
ARG (`eqx.filter_jit` traces the array leaves). Never close over the model in a jit: a
frozen 8B target captured as a constant bakes multi-GB weights into the HLO.
"""

from typing import Any, Protocol, runtime_checkable

import jax.numpy as jnp
from jax import random as jax_random
from jax.sharding import Mesh
from jaxtyping import Array, Bool, Float

from param_decomp.components import DecompVU, SiteSpec

SiteMasks = dict[str, Float[Array, "*leading C"]]
SiteDeltaMasks = dict[str, Float[Array, "*leading"]]
SiteRoutes = dict[str, Bool[Array, "*leading"]] | None
"""Per-site per-position routing; `None` routes every position to the decomposition
(SPEC §1.3). Positions routing False take the frozen `x @ W` path."""


def all_false_routes(site_names: tuple[str, ...], leading: tuple[int, ...]) -> dict[str, Array]:
    """Route every position to the frozen path so `masked_site_outputs` returns the
    target `x @ W` per site (the clean recon target used by the hidden-acts / attn-pattern
    eval metrics)."""
    return {s: jnp.zeros(leading, bool) for s in site_names}


@runtime_checkable
class DecomposedModel(Protocol):
    """The interface a vendored target implements for the generic trainer (see module
    docstring). The concrete impl is an `eqx.Module` carrying its FROZEN target weights as
    array fields — so it threads into the jitted step as a pytree arg (array leaves traced,
    static fields baked), never as a closed-over multi-GB constant. The methods below take
    only the *runtime-varying* arguments; the frozen weights ride on `self`.

    The TRAINABLE V/U (`vu`) stay an explicit method arg, NOT a `self` field: they have a
    different lifecycle (fp32 masters, their own optimizer + checkpoint, C-sharded while the
    frozen weights replicate), and the step casts/details them independently of the model.

    `sites` fixes the canonical site order — chunking (SPEC S10) and the CI fn's
    input/output concatenation both follow it.

    `leading_axes` names the position axes of the activation waist (batch is implicit and
    always present): `("sequence",)` for an LM, `()` for a TMS-style target. At trainer
    construction it must equal the CI fn's `expects_axes` (early fail) so the CI fn stays
    per-domain without the generic loop adapting.

    `recon_loss_fn(masked_output, clean_output) -> scalar` is the recon comparison the step
    minimizes (SPEC §2.3): the LM uses `kl_per_position`; a non-LM target supplies its own
    (e.g. MSE/geometric). It must reduce to a scalar and contract whatever shape the model
    emits. It is a static method/attr on the concrete model, not a forward over `self`.
    """

    sites: tuple[SiteSpec, ...]
    leading_axes: tuple[str, ...]

    @property
    def site_names(self) -> tuple[str, ...]: ...

    def shardings(self, mesh: Mesh) -> "DecomposedModel":
        """Per-leaf `dp` placement of the frozen target weights, matching this model's
        pytree structure (each array leaf → a `NamedSharding`). The frozen target is
        REPLICATED on every device (small relative to activations; replicating avoids
        all-gathering it every forward). Applied via `jax.jit(..., out_shardings=...)`."""
        ...

    def recon_loss_fn(self, masked_output: Any, clean_output: Any) -> Float[Array, ""]:
        """Recon comparison the step minimizes (SPEC §2.3). LM: `kl_per_position`."""
        ...

    def clean_output(self, inputs: Any, /) -> Any:
        """All-frozen forward — the recon target (SPEC S3); never the `mask=1` decomposed
        identity. `inputs` is positional-only and target-specific (an LM's token ids → embed;
        a toy's feature vector, which already is the waist). `Any` return: an LM emits
        `[*leading, vocab]` logits, a bio target a tuple of heads or coordinates."""
        ...

    def read_activations(
        self, inputs: Any, /, wanted: tuple[str, ...]
    ) -> dict[str, Float[Array, "*leading d_tap"]]:
        """The CI fn's activation accessor. `wanted` is the CI fn's static `input_names` —
        OPAQUE keys the target knows how to produce (an LM's `resid.{layer}` taps; a
        positionless toy's per-site inputs). The target is the only key→activation
        interpreter; core just routes by key."""
        ...

    def prepare_compute_weights(self, vu: DecompVU) -> Any:
        """Build the per-step COMPUTE weights from the fp32 ÷N master `vu`, ONCE per step
        (SPEC unchanged — a read-only compute view). Mask-INDEPENDENT, so the result is shared
        by every forward in the step: the engine calls this once and passes the result as the
        `prepared` arg to all `masked_output` / `masked_site_outputs` calls. For a sharded LM
        target this is where the ÷N→÷fsdp cross-node gather (+ bf16 cast) happens — once, off
        the hot path, instead of once per forward. For an unsharded toy it is identity (returns
        `vu`). The opaque return type is the target's own — only it consumes it."""
        ...

    def masked_output(
        self,
        prepared: Any,
        inputs: Any,
        /,
        masks: SiteMasks,
        delta_masks: SiteDeltaMasks,
        routes: SiteRoutes,
        live: tuple[str, ...],
        has_delta: bool,
        *,
        remat: bool,
    ) -> Any:
        """The masked decomposed forward (SPEC §1.3, S2). `prepared` is the output of
        `prepare_compute_weights` (the shared per-step compute weights). `live` (static under
        jit) lists the sites running their decomposed forward; all other sites run the frozen
        `x @ W` path. `masks`/`delta_masks` may broadcast over the batch dim (the PPGD source
        case). `has_delta` (static) False skips the `x @ Δ` matmul for constant-source entries
        whose delta mask is a constant 0 (LOSS_PARITY_DESIGN §4b). `remat` (static) gates
        gradient-checkpointing the forward at the model's natural granularity (a deep target
        rematerializes per-layer, recomputing one layer at a time in the backward instead of
        storing every layer's activations — the dominant step-memory term at depth)."""
        ...

    def stack_ci(self, ci_lower: dict[str, Array]) -> Any:
        """Build the target-specific SHARED CI form (opaque, like `prepare_compute_weights`):
        the CI envelope reshaped ONCE per step so the stochastic recon forwards can reuse it.
        A scan target stacks the per-site CI into its per-layer scan layout (shared, so N
        forwards' mask stacks collapse to one — the memory win); a target without that structure
        returns `ci_lower` unchanged. Consumed only by `masked_output_stochastic`."""
        ...

    def masked_output_stochastic(
        self,
        prepared: Any,
        inputs: Any,
        /,
        ci_stacked: Any,
        draw_key: Array,
        routes: SiteRoutes,
        live: tuple[str, ...],
        has_delta: bool,
        *,
        remat: bool,
    ) -> Any:
        """Stochastic recon forward that builds masks INTERNALLY from the shared `ci_stacked`
        (`= stack_ci(ci.lower)`) + per-position draws from `draw_key`, rather than taking
        pre-built masks. A scan target RECOMPUTES the masks inside its checkpointed block (the
        per-forward mask is never held — a memory win, faithful by checkpoint determinism);
        targets without that structure delegate to `run_stochastic_masked_output` (build masks,
        then `masked_output`). Same forward semantics as `masked_output` with stochastic sources;
        the random draw need not match any other path's (only fwd==bwd faithfulness matters)."""
        ...

    def masked_site_outputs(
        self,
        prepared: Any,
        inputs: Any,
        /,
        masks: SiteMasks,
        delta_masks: SiteDeltaMasks,
        routes: SiteRoutes,
        live: tuple[str, ...],
        has_delta: bool,
    ) -> dict[str, Float[Array, "*leading d_out"]]:
        """Per-`live`-site decomposed LINEAR OUTPUT of `masked_output`'s forward
        (`((x@V)*m)@U + (x@Δ)*d`), keyed by site (SPEC S31). `prepared` is the output of
        `prepare_compute_weights`. For the offline hidden-acts recon eval metrics only — never
        the recon grid, which stays KL-on-final-logits."""
        ...

    def weight_deltas(self, vu: DecompVU) -> dict[str, Float[Array, "d_out d_in"]]:
        """fp32 `W − V@U` per site from the fp32 master `vu` (SPEC N2)."""
        ...


def stochastic_site_masks(
    ci_lower: dict[str, Array], live: tuple[str, ...], draw_key: Array, has_delta: bool
) -> tuple[dict[str, Array], dict[str, Array]]:
    """Per-live-site stochastic masks `ci + (1−ci)·U[0,1]` (+ delta `U[0,1]`), keyed per site
    from `draw_key`. The generic stochastic-source formula (SPEC S12 stochastic), shared by the
    non-recompute `masked_output_stochastic` fallback."""
    mask_key, delta_key = jax_random.split(draw_key)
    masks: dict[str, Array] = {}
    delta_masks: dict[str, Array] = {}
    for site_idx, site in enumerate(live):
        ci = ci_lower[site]
        masks[site] = ci + (1.0 - ci) * jax_random.uniform(
            jax_random.fold_in(mask_key, site_idx), ci.shape, ci.dtype
        )
        if has_delta:
            delta_masks[site] = jax_random.uniform(
                jax_random.fold_in(delta_key, site_idx), ci.shape[:-1], ci.dtype
            )
    return masks, delta_masks


def run_stochastic_masked_output(
    model: "DecomposedModel",
    prepared: Any,
    inputs: Any,
    ci_lower: dict[str, Array],
    draw_key: Array,
    routes: SiteRoutes,
    live: tuple[str, ...],
    has_delta: bool,
    *,
    remat: bool,
) -> Any:
    """Default `masked_output_stochastic` for targets without an in-block recompute: build the
    stochastic masks here (no shared-CI reuse — `stack_ci` was identity), then run the standard
    `masked_output`. Correct + uniform; only scan targets override for the memory win."""
    masks, delta_masks = stochastic_site_masks(ci_lower, live, draw_key, has_delta)
    return model.masked_output(
        prepared, inputs, masks, delta_masks, routes, live, has_delta, remat=remat
    )


def chunk_sites(site_names: tuple[str, ...], sites_per_chunk: int) -> tuple[tuple[str, ...], ...]:
    """Sequential `sites_per_chunk`-groups in the canonical site order (SPEC S10)."""
    assert len(site_names) % sites_per_chunk == 0, (
        f"{len(site_names)} sites not divisible by sites_per_chunk={sites_per_chunk}"
    )
    return tuple(
        tuple(site_names[i : i + sites_per_chunk])
        for i in range(0, len(site_names), sites_per_chunk)
    )
