"""CPU tests for per-site faithfulness observability.

The load-bearing invariant: the per-site `‖Δ_s‖²_F` this step emits are the SAME per-site
deltas the global `FaithfulnessLoss` reduces, so `Σ_s result / Σ_s numel == FaithfulnessLoss`
exactly. Also checks the zeroed-vu `‖W_s‖²_F` denominator and the rank-indexed scalar helper.
"""

import jax
import jax.numpy as jnp

from param_decomp.components import init_decomp_vu
from param_decomp.losses import faithfulness_loss
from param_decomp.per_site_faith_eval import make_per_site_faith_step, per_site_faith_scalars
from param_decomp.targets.llama8b import llama_site_specs, mlp_family_site_cs
from param_decomp.tests.test_llama8b import _tiny_cfg, _tiny_decomposed_lm


def _tiny_lm_and_vu():
    cfg = _tiny_cfg()
    sites = llama_site_specs(cfg, mlp_family_site_cs(4, 5, 8))
    lm = _tiny_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))
    vu = init_decomp_vu(sites, jax.random.PRNGKey(1))
    return lm, vu


def test_per_site_delta_sq_matches_weight_deltas():
    lm, vu = _tiny_lm_and_vu()
    delta_sq = make_per_site_faith_step(lm)(lm, vu)
    deltas = lm.weight_deltas(vu)
    assert set(delta_sq) == set(deltas)
    for site, delta in deltas.items():
        expected = (delta.astype(jnp.float32) ** 2).sum()
        assert jnp.allclose(delta_sq[site], expected, rtol=1e-6)


def test_sum_matches_global_faithfulness_loss():
    lm, vu = _tiny_lm_and_vu()
    delta_sq = make_per_site_faith_step(lm)(lm, vu)
    deltas = lm.weight_deltas(vu)
    total_numel = sum(d.size for d in deltas.values())
    reconstructed = sum(float(v) for v in delta_sq.values()) / total_numel
    assert jnp.allclose(reconstructed, faithfulness_loss(deltas), rtol=1e-6)


def test_zeroed_vu_gives_frozen_weight_norms():
    lm, vu = _tiny_lm_and_vu()
    step = make_per_site_faith_step(lm)
    zero_vu = jax.tree.map(jnp.zeros_like, vu)
    w_sq = step(lm, zero_vu)
    for site, w in lm.weight_deltas(zero_vu).items():  # Δ = W − 0 = W
        assert jnp.allclose(w_sq[site], (w.astype(jnp.float32) ** 2).sum(), rtol=1e-6)
        assert float(w_sq[site]) > 0.0


def test_per_site_faith_scalars_ranks_and_keys():
    rel_frob = {"a": 0.1, "b": 0.9, "c": 0.3, "d": 0.5}
    abs_frob = {"a": 2.0, "b": 9.0, "c": 3.0, "d": 5.0}
    out = per_site_faith_scalars(rel_frob, abs_frob, top_k=3)
    assert out["eval/faith/rel_frob_top1"] == 0.9
    assert out["eval/faith/rel_frob_top2"] == 0.5
    assert out["eval/faith/rel_frob_top3"] == 0.3
    assert "eval/faith/rel_frob_top4" not in out
    assert out["eval/faith/abs_frob_max"] == 9.0


def test_per_site_faith_scalars_top_k_exceeds_site_count():
    rel_frob = {"a": 0.2, "b": 0.7}
    abs_frob = {"a": 1.0, "b": 4.0}
    out = per_site_faith_scalars(rel_frob, abs_frob, top_k=8)
    assert out["eval/faith/rel_frob_top1"] == 0.7
    assert out["eval/faith/rel_frob_top2"] == 0.2
    assert "eval/faith/rel_frob_top3" not in out
