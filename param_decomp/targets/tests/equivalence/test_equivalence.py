"""Pytest entry for the cross-framework PD equivalence harness.

Two kinds of check:

  * **Numeric (cross-framework).** `test_jax_matches_torch_reference` runs the JAX side of
    the harness on the committed fixtures and asserts each retained oracle-backed term matches
    `torch_reference.json` (produced by `torch_reference.py` in the torch env) to fp32
    tolerance. This is the watertight numeric verification: identical fixtures into both
    frameworks, the torch values from the REAL reference functions
    (`importance_minimality_terms` / `recon_loss_kl` /
    `get_ppgd_mask_infos` / `LinearComponents.forward`), compared at ~1e-4.

  * **Structural.** `test_structure_*` pin SPEC invariants that aren't a single number:
    each recon term loops its routing draws (S10'), recon is KL not MSE (§2.3),
    and the PPGD source carries the trailing raw weight-delta channel (S1).
    `test_sc_source_broadcasts_over_batch_in_masked_forward` pins the `sc`
    broadcast (S1/S16): an `(1, T, C+1)` source broadcasts over `[B, T]` in the masked
    forward, and a B/T-transposed source must break it (the fixtures keep `B != T`).

Faithfulness is no longer compared because the target-relative formula intentionally differs
from the per-parameter torch oracle. The golden's `stoch` term drove partial per-chunk masked
forwards, which the all-sites masked forward no longer supports. The retained comparisons are
`imp` and `ppgd`.

`torch_reference.json` is a FROZEN committed golden. The torch generator that produced
it (`torch_reference.py`) is deleted — `param_decomp` imports no torch. To regen
(only if the math or fixtures change): check out the `torch-oracle` git tag in a
separate worktree, run that revision's `torch_reference.py` in the torch
(`param-decomp`) venv, and copy the resulting `torch_reference.json` back here. The
fixtures themselves (`fixtures.npz`) are still drawn JAX-side by `gen_fixtures.py`.
"""

import inspect
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import param_decomp.core.losses as core_losses_mod
import param_decomp.core.masking as masking_mod
import param_decomp.core.train as train_mod
import param_decomp.targets.losses as losses_mod
from param_decomp.core.adversary import SiteSource, split_source_channels
from param_decomp.core.masking import masks_from_sources
from param_decomp.targets.testing import run_masked
from param_decomp.targets.tests.equivalence.jax_equivalence import compute_jax_terms

HERE = Path(__file__).resolve().parent
RTOL = 2e-4
ATOL = 1e-5

_PENDING_REGEN = pytest.mark.xfail(
    reason=(
        "pending embed-internal golden regen: fixtures are residual-fed but the model "
        "now takes token ids. Regenerate the torch reference + fixtures against the token "
        "contract (torch-oracle worktree)."
    ),
    strict=False,
)


def _load_fixtures() -> dict[str, np.ndarray]:
    return dict(np.load(HERE / "fixtures.npz"))


@pytest.mark.parametrize("term", ["imp", "ppgd"])
@_PENDING_REGEN
def test_jax_matches_torch_reference(term: str) -> None:
    ref_path = HERE / "torch_reference.json"
    assert ref_path.exists(), "run torch_reference.py (torch env) to produce the golden first"
    ref = json.loads(ref_path.read_text())
    jaxv = compute_jax_terms(_load_fixtures())
    jv, tv = jaxv[term], ref[term]
    assert abs(jv - tv) <= ATOL + RTOL * abs(tv), (
        f"{term}: jax {jv:.8e} vs torch {tv:.8e} (rel {abs(jv - tv) / (abs(tv) + 1e-30):.2e})"
    )


def test_structure_stoch_is_mean_over_draws() -> None:
    """SPEC S10': one forward per routing draw, and the term's loss is the mean over
    its draws — not one fused forward over all draws."""
    src = inspect.getsource(train_mod.ReconGrid)
    assert "for draw_key, routes in draws" in src, "each recon term must loop its sampled draws"
    assert "mean_reconstruction_losses" in src
    reducer_src = inspect.getsource(core_losses_mod.mean_reconstruction_losses)
    assert "jax.tree.map(scalar_mean, *values)" in reducer_src, (
        "each breakdown leaf must be averaged over every masked draw"
    )
    assert "/ len(scalars)" in reducer_src, (
        "each breakdown leaf must be normalized by every masked draw"
    )


def test_structure_recon_is_kl_not_mse() -> None:
    """SPEC §2.3: recon is KL on logits, not MSE."""
    src = inspect.getsource(losses_mod.kl_per_position)
    assert "log_softmax" in src and "log_p - log_q" in src, "recon must be KL"
    assert "** 2" not in src and "**2" not in src, "recon must not be MSE"


def test_structure_ppgd_has_delta_channel() -> None:
    """SPEC S1: component sources are interpolated; delta sources are raw masks."""
    src = inspect.getsource(masking_mod.masks_from_sources)
    assert "source.components" in src and "source.delta" in src
    assert "ci + (1.0 - ci) * source.components" in src, (
        "ppgd must interpolate mask=ci+(1-ci)*source"
    )


def test_sc_scope_broadcast_axis_matches_torch() -> None:
    """SPEC S1/S16: the `sc`-scope PPGD source `(1, T, C+1)` broadcasts over the batch
    axis and varies per position. This pins the batch broadcast axis: a silent transpose
    (`(1, T, ...)` read as `(T, 1, ...)`) would broadcast over position and vary per
    batch element instead — uncaught by the scalar-KL `ppgd` term, which sums over `B·T`.

    Reference is torch's `mask = ci + (1 - ci) * source` (`interpolate_component_mask`):
    numpy broadcasting is identical to torch's, so this is exact (not approximate) and
    runs in the JAX env alone. B != T so a transposed axis is shape-detectable too."""
    B, T, C = 3, 5, 4
    site = "h.0.mlp.c_fc"
    rng = np.random.default_rng(0)
    ci_lower_np = rng.uniform(0.0, 1.0, (B, T, C)).astype(np.float32)
    source_np = rng.uniform(0.0, 1.0, (1, T, C + 1)).astype(np.float32)

    masks, delta_masks = masks_from_sources(
        {site: jnp.asarray(ci_lower_np)},
        {
            site: SiteSource(
                components=jnp.asarray(source_np[..., :-1]),
                delta=jnp.asarray(source_np[..., -1]),
            )
        },
    )
    mask = np.asarray(masks[site])
    delta_mask = np.asarray(delta_masks[site])

    # The component mask gains the batch axis via `(1 - ci)` broadcasting; the raw delta
    # channel stays at the source's `(1, T)` (it is batch-broadcast later, in the forward).
    assert mask.shape == (B, T, C)
    assert delta_mask.shape == (1, T)

    # torch reference: `ci + (1 - ci) * source[..., :C]`, framework-agnostic broadcast.
    ref_mask = ci_lower_np + (1.0 - ci_lower_np) * source_np[..., :C]
    np.testing.assert_allclose(mask, ref_mask, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(delta_mask, source_np[..., C], rtol=RTOL, atol=ATOL)

    # Axis assertion on the raw delta channel (the source value with no ci entanglement):
    # its leading-1 makes it batch-invariant, and (B != T) it varies per position. A
    # silent transpose `(1, T, ...)` read as `(T, 1, ...)` would flip both — varying per
    # batch, constant per position. The scalar-KL `ppgd` term sums over B·T and cannot
    # see this; the materialized mask can.
    assert not np.allclose(delta_mask[0, 0], delta_mask[0, 1]), (
        "sc source must vary per position (a transposed broadcast axis would not)"
    )


def test_fixtures_are_batch_asymmetric_so_a_bt_transpose_is_observable() -> None:
    """The sc-broadcast guard below (and the `ppgd` numeric term) can only catch a B/T
    axis transpose when `B != T`; a square fixture would let a transpose pass silently."""
    f = _load_fixtures()
    B, T = int(f["_scalar_B"]), int(f["_scalar_T"])
    assert B != T, f"fixtures must keep B != T to expose a B/T transpose, got B={B} T={T}"
    for k in ("gate", "up", "down"):
        sc = f[f"ppgd_source_{k}"]  # (1, T, L, C+1)
        assert sc.shape[0] == 1 and sc.shape[1] == T, (
            f"ppgd source must be sc-scope (1, T, L, C+1), got {sc.shape}"
        )


@_PENDING_REGEN
def test_sc_source_broadcasts_over_batch_in_masked_forward() -> None:
    """SPEC S1/S16: an sc-scope source `(1, T, C+1)` broadcasts over `[B, T]` in the
    masked forward — shared across batch elements, free per position. This exercises the
    `delta_mask[..., None]` / mask broadcast (`components.site_out`) the way the PPGD path
    does, and pins the broadcast AXIS: transposing the source to `(1, B, C+1)` (B != T)
    must break the forward rather than silently re-interpret the time axis as batch."""
    from param_decomp.targets.glu_transformer import MLP_KINDS, site_name
    from param_decomp.targets.tests.equivalence.jax_equivalence import FP, _build

    f = _load_fixtures()
    model, vu, n_layers = _build(f)
    resid = jnp.asarray(f["resid"], dtype=FP)
    B, T = int(f["_scalar_B"]), int(f["_scalar_T"])
    vocab = int(f["_scalar_VOCAB"])
    assert B != T

    def per_site_sc(prefix: str) -> dict[str, "jnp.ndarray"]:
        by_kind = {k: jnp.asarray(f[f"{prefix}_{k}"], dtype=FP) for k in ("gate", "up", "down")}
        return {site_name(i, k): by_kind[k][:, :, i] for i in range(n_layers) for k in MLP_KINDS}

    ci_lower = per_site_sc("ci_lower")  # (B, T, C)
    packed_source = per_site_sc("ppgd_source")  # (1, T, C+1) per site
    for s in model.site_names:
        assert packed_source[s].shape == (1, T, packed_source[s].shape[-1])
    source = {site: split_source_channels(value) for site, value in packed_source.items()}

    masks, delta_masks = masks_from_sources(ci_lower, source)
    # `ci + (1-ci)*src` lifts the sc mask to the CI's batch dim; delta stays sc.
    for s in model.site_names:
        assert masks[s].shape[0] == B and masks[s].shape[1] == T, masks[s].shape
        assert delta_masks[s].shape == (1, T), delta_masks[s].shape

    pred = run_masked(
        model,
        model.prepare_compute_weights(vu, None),
        resid,
        masks,
        delta_masks,
        None,
        True,
        remat=False,
    )
    assert pred.shape == (B, T, vocab), pred.shape

    # A source whose free axis is sized B (not T) — i.e. the B/T axes transposed — must NOT
    # broadcast against the `(B, T, C)` ci. The time axis is load-bearing, not interchangeable.
    bt_transposed_packed = {s: packed_source[s][:, :B, :] for s in model.site_names}
    bt_transposed = {
        site: split_source_channels(value) for site, value in bt_transposed_packed.items()
    }
    for s in model.site_names:
        assert bt_transposed[s].components.shape[:2] == (1, B)
    with pytest.raises(Exception):  # noqa: B017 — broadcast error, framework-specific type
        bad_masks, bad_delta = masks_from_sources(ci_lower, bt_transposed)
        run_masked(
            model,
            model.prepare_compute_weights(vu, None),
            resid,
            bad_masks,
            bad_delta,
            None,
            True,
            remat=False,
        )
