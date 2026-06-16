"""Pytest entry for the cross-framework PD equivalence harness.

Two kinds of check:

  * **Numeric (cross-framework).** `test_jax_matches_torch_reference` runs the JAX side of
    the harness on the committed fixtures and asserts every loss term matches the committed
    `torch_reference.json` (produced by `torch_reference.py` in the torch env) to fp32
    tolerance. This is the watertight numeric verification: identical fixtures into both
    frameworks, the torch values from the REAL reference functions
    (`faithfulness_loss` / `importance_minimality_terms` / `recon_loss_kl` /
    `get_ppgd_mask_infos` / `LinearComponents.forward`), compared at ~1e-4.

  * **Structural.** `test_structure_*` pin SPEC invariants that aren't a single number:
    the stochastic recon runs ONE forward PER CHUNK (S10), recon is KL not MSE (§2.3),
    and the PPGD source carries the trailing raw weight-delta channel (S1).

Regenerate the cross-framework golden (only needed if the math or fixtures change):

    # JAX env:
    python jax_single_pool/tests/equivalence/gen_fixtures.py
    # torch (param-decomp) env:
    python jax_single_pool/tests/equivalence/torch_reference.py
"""

import inspect
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import jax_single_pool.adversary as adversary_mod
import jax_single_pool.losses as losses_mod
import jax_single_pool.train as train_mod
from jax_single_pool.adversary import source_masks
from jax_single_pool.tests.equivalence.jax_equivalence import compute_jax_terms

HERE = Path(__file__).resolve().parent
RTOL = 2e-4
ATOL = 1e-5


@pytest.mark.parametrize("term", ["faith", "imp", "stoch", "ppgd"])
def test_jax_matches_torch_reference(term: str) -> None:
    ref_path = HERE / "torch_reference.json"
    assert ref_path.exists(), "run torch_reference.py (torch env) to produce the golden first"
    ref = json.loads(ref_path.read_text())
    jaxv = compute_jax_terms(dict(np.load(HERE / "fixtures.npz")))
    jv, tv = jaxv[term], ref[term]
    assert abs(jv - tv) <= ATOL + RTOL * abs(tv), (
        f"{term}: jax {jv:.8e} vs torch {tv:.8e} (rel {abs(jv - tv) / (abs(tv) + 1e-30):.2e})"
    )


def test_structure_stoch_is_per_chunk() -> None:
    """SPEC S10: one forward per (chunk, sample), normalized by `n_chunks · n_samples`
    — matching the torch chunkwise pool, not one fused forward over all sites."""
    src = inspect.getsource(train_mod.make_train_step)
    assert "for entry_idx, entry in enumerate(term.plan)" in src, (
        "each recon term must loop its plan's entries"
    )
    assert "/ n_forwards" in src, "each term must average over ALL forwards (every draw)"


def test_structure_recon_is_kl_not_mse() -> None:
    """SPEC §2.3: recon is KL on logits, not MSE."""
    src = inspect.getsource(losses_mod.kl_per_position)
    assert "log_softmax" in src and "log_p - log_q" in src, "recon must be KL"
    assert "** 2" not in src and "**2" not in src, "recon must not be MSE"


def test_structure_ppgd_has_delta_channel() -> None:
    """SPEC S1: PPGD masks interpolate `ci + (1-ci)*src[:, :C]`; the trailing channel
    is the raw weight-delta mask (no ci interpolation)."""
    src = inspect.getsource(adversary_mod.source_masks)
    assert "[..., :-1]" in src and "[..., -1]" in src, "ppgd source needs the delta channel"
    assert "ci_lower[site] + (1.0 - ci_lower[site]) * source[..., :-1]" in src, (
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

    masks, delta_masks = source_masks(
        {site: jnp.asarray(ci_lower_np)}, {site: jnp.asarray(source_np)}, (site,)
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
