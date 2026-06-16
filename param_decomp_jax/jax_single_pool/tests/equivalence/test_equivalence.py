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

import numpy as np
import pytest

import jax_single_pool.adversary as adversary_mod
import jax_single_pool.losses as losses_mod
import jax_single_pool.train as train_mod
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


def test_sigmoid_parameterization_matches_torch_effective_sources() -> None:
    """SPEC S15/S16, §6 `PROJ`/`EFFECTIVE` (sigmoid variant). Torch's
    `get_effective_sources` applies `sigmoid` to the WHOLE latent (incl. the trailing
    delta channel) before `get_ppgd_mask_infos` interpolates
    `mask = ci + (1-ci)*EFFECTIVE(src)[..., :C]` / `delta = EFFECTIVE(src)[..., -1]`.
    `source_masks(..., "sigmoid")` must reproduce that math from the SAME unbounded
    latent (the leaf the ascent updates without projection)."""
    import jax
    import jax.numpy as jnp

    key = jax.random.PRNGKey(0)
    ci_key, src_key = jax.random.split(key)
    sites = ("a", "b")
    B, T, C = 2, 3, 4
    ci_lower = {
        s: jax.random.uniform(jax.random.fold_in(ci_key, i), (B, T, C)) for i, s in enumerate(sites)
    }
    latent = {
        s: jax.random.normal(jax.random.fold_in(src_key, i), (1, T, C + 1)) * 3.0
        for i, s in enumerate(sites)
    }

    masks, delta_masks = adversary_mod.source_masks(ci_lower, latent, sites, "sigmoid")
    for s in sites:
        effective = jax.nn.sigmoid(latent[s])
        expected_mask = ci_lower[s] + (1.0 - ci_lower[s]) * effective[..., :-1]
        np.testing.assert_allclose(
            np.asarray(masks[s]), np.asarray(expected_mask), rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            np.asarray(delta_masks[s]), np.asarray(effective[..., -1]), rtol=1e-6, atol=1e-6
        )
        assert jnp.all((masks[s] >= 0.0) & (masks[s] <= 1.0)), "sigmoid masks must stay in [0,1]"


def test_sigmoid_parameterization_init_is_normal_and_proj_is_identity() -> None:
    """SPEC §6: sigmoid init is N(0,1) (unbounded latent, range exceeds [0,1]); `PROJ`
    is identity so an ascent that drives a source past 1 is NOT clamped back."""
    import jax
    import jax.numpy as jnp

    sites = ("a",)
    src = adversary_mod.init_persistent_sources(sites, (8,), 16, "sigmoid", jax.random.PRNGKey(1))
    assert jnp.max(src["a"]) > 1.0 and jnp.min(src["a"]) < 0.0, "N(0,1) init must escape [0,1]"

    from param_decomp_config.schedule import ScheduleConfig

    adam = adversary_mod.AdamPGDConfig(
        lr_schedule=ScheduleConfig(fn_type="constant", start_val=1.0)
    )
    state = adversary_mod.init_sources_adam_state(src)
    grad = {"a": jnp.ones_like(src["a"])}
    ascended, _ = adversary_mod.sources_adam_ascend_project(
        src, grad, state, jnp.asarray(50.0), adam, "sigmoid"
    )
    assert jnp.max(ascended["a"]) > 1.0, "sigmoid PROJ must be identity (no clamp to [0,1])"
