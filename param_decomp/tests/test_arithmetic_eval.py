"""CPU tests for the arithmetic-grid eval (`arithmetic_eval.py`).

Pins: the fused step returns per-component CI AND activation `x@V` at the answer position
with the batch axis kept (vs hand-rolled readouts); row-major `(a, b)` reshape; active-set
selection (max CI > threshold, descending); n_alive / n_dropped scalars; and the renderer
emits valid CI + activation PNGs over the shared active set.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from param_decomp.arithmetic_eval import (
    ArithmeticGrid,
    ComponentActivationModel,
    accumulate_arithmetic_grids,
    active_components,
    make_arithmetic_grid_step,
    n_alive_scalars,
    plot_component_grids,
    render_arithmetic_figures,
    select_active,
)
from param_decomp.ci_fn import lower_leaky_hard_sigmoid
from param_decomp.components import init_decomp_vu
from param_decomp.targets.llama8b import llama_site_specs, mlp_family_site_cs
from param_decomp.tests.test_llama8b import _tiny_cfg, _tiny_decomposed_lm
from param_decomp.tests.test_slow_eval import _build_ci_fn
from param_decomp.train import COMPUTE_DT, cast_floating

N_A, N_B = 3, 4
T = 5
ANSWER_POSITION = T - 1
SITE = "layers.4.mlp.gate_proj"


def _tiny_setup():
    cfg = _tiny_cfg()
    C = 8
    sites = llama_site_specs(cfg, mlp_family_site_cs(4, 5, C))
    lm = _tiny_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))
    ci_fn = _build_ci_fn(lm, cfg.n_embd, jax.random.PRNGKey(2))
    return cfg, lm, ci_fn, C


def _grid() -> ArithmeticGrid:
    return ArithmeticGrid(a_values=tuple(range(N_A)), b_values=tuple(range(N_B)), symbol="+")


def test_grid_step_ci_and_xv_match_hand_rolled():
    cfg, lm, ci_fn, C = _tiny_setup()
    assert isinstance(lm, ComponentActivationModel)
    vu = init_decomp_vu(lm.sites, jax.random.PRNGKey(1))
    tokens = jax.random.randint(jax.random.PRNGKey(4), (N_A * N_B, T), 0, cfg.vocab_size)
    step = make_arithmetic_grid_step(lm, ANSWER_POSITION)
    ci_grids, xv_grids = accumulate_arithmetic_grids(step, lm, vu, ci_fn, [tokens], N_A * N_B)

    names = lm.site_names
    # CI hand-roll: bf16 readout, slice the answer position.
    ci_fn_bf16 = cast_floating(ci_fn, COMPUTE_DT)
    taps = {
        k: x.astype(COMPUTE_DT) for k, x in lm.read_activations(tokens, ci_fn.input_names).items()
    }
    logits = {s: v.astype("float32") for s, v in ci_fn_bf16(taps, remat=False).logits.items()}
    # xV hand-roll: all-ones masks -> site output == (x@V) @ U, so x@V projected through U
    # reproduces masked_site_outputs at the answer position.
    prepared = lm.prepare_compute_weights(cast_floating(vu, COMPUTE_DT))
    leading = tokens.shape
    ones = {s: jnp.ones((*leading, C), COMPUTE_DT) for s in names}
    zeros_delta = {s: jnp.zeros(leading, COMPUTE_DT) for s in names}
    outputs = lm.masked_site_outputs(prepared, tokens, ones, zeros_delta, None, names, False)
    for site in names:
        ci_exp = np.asarray(lower_leaky_hard_sigmoid(logits[site]))[:, ANSWER_POSITION, :]
        assert ci_grids[site].shape == (N_A * N_B, C)
        assert ci_grids[site].min() >= 0.0 and ci_grids[site].max() <= 1.0
        np.testing.assert_allclose(ci_grids[site], ci_exp, rtol=1e-4, atol=1e-4)
        xv = xv_grids[site]
        assert xv.shape == (N_A * N_B, C) and np.all(np.isfinite(xv))
        _, u = vu.vu[site]
        out_got = np.asarray(outputs[site])[:, ANSWER_POSITION, :].astype(np.float32)
        np.testing.assert_allclose(out_got, xv @ np.asarray(u, np.float32), atol=1e-2)


def test_to_grid_is_row_major_a_then_b():
    grid = _grid()
    per_prompt = np.array([[i * 10 + j] for i in range(N_A) for j in range(N_B)], dtype=float)
    reshaped = grid.to_grid(per_prompt)  # (N_A, N_B, 1)
    for i in range(N_A):
        for j in range(N_B):
            assert reshaped[i, j, 0] == i * 10 + j


def test_active_components_selects_above_threshold_ordered_by_max():
    ci = np.array([[0.05, 0.9, 0.3, 0.2], [0.0, 0.1, 0.3, 0.95]])  # max = [0.05,0.9,0.3,0.95]
    assert active_components(ci, threshold=0.1).tolist() == [3, 1, 2]  # 0 below thr; rest by max
    assert active_components(np.full((2, 5), 0.05), threshold=0.1).size == 0


def test_select_active_and_n_alive_scalars():
    ci_grids = {
        SITE: np.array([[0.9, 0.0, 0.0], [0.0, 0.2, 0.0]]),  # 2 alive
        "layers.4.mlp.up_proj": np.array([[0.05, 0.05], [0.0, 0.0]]),  # 0 alive
    }
    active = select_active(ci_grids, (0.1,))
    assert active[0.1][SITE].tolist() == [0, 1]
    assert active[0.1]["layers.4.mlp.up_proj"].size == 0
    scalars = n_alive_scalars(active, top_k=64)
    assert scalars["n_alive/thr0.1/" + SITE] == 2.0
    assert scalars["n_alive/thr0.1/layers.4.mlp.up_proj"] == 0.0
    assert scalars["n_alive/thr0.1/total"] == 2.0


def test_n_alive_scalars_reports_dropped_beyond_top_k():
    ci = np.zeros((4, 5))
    ci[:, [0, 1, 2]] = 0.8  # 3 alive
    scalars = n_alive_scalars(select_active({SITE: ci}, (0.1,)), top_k=2)
    assert scalars["n_alive/thr0.1/" + SITE] == 3.0
    assert scalars["n_dropped/thr0.1/" + SITE] == 1.0  # 3 alive - top_k 2


def test_plot_component_grids_png_and_rejects_empty():
    grid = _grid()
    ci = np.zeros((N_A * N_B, 4))
    ci[:, 1] = 0.8
    assert (
        plot_component_grids(ci, grid, np.array([1]), "t", "viridis", (0.0, 1.0))[:4] == b"\x89PNG"
    )
    signed = np.linspace(-2, 2, N_A * N_B * 4).reshape(N_A * N_B, 4)  # auto-symmetric range
    assert (
        plot_component_grids(signed, grid, np.array([0, 1]), "t", "coolwarm", None)[:4]
        == b"\x89PNG"
    )
    with pytest.raises(AssertionError):  # empty selection is a caller bug, not a silent no-op
        plot_component_grids(ci, grid, np.array([], dtype=int), "t", "viridis", (0.0, 1.0))


def test_render_arithmetic_figures_pairs_ci_and_activation():
    grid = _grid()
    ci = np.zeros((N_A * N_B, 5))
    ci[:, [0, 1, 2]] = 0.8
    xv = np.random.default_rng(0).standard_normal((N_A * N_B, 5))
    active = select_active({SITE: ci}, (0.1, 0.5))
    figures = render_arithmetic_figures({SITE: ci}, {SITE: xv}, active, grid, top_k=2)
    assert set(figures) == {
        f"figures/ci_grid/thr0.1/{SITE}",
        f"figures/activation_grid/thr0.1/{SITE}",
        f"figures/ci_grid/thr0.5/{SITE}",
        f"figures/activation_grid/thr0.5/{SITE}",
    }
    for png in figures.values():
        assert png[:4] == b"\x89PNG"
