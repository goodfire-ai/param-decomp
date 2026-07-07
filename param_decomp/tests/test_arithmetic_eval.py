"""CPU tests for the arithmetic-grid eval (`arithmetic_eval.py`).

Pins: the fused step returns per-component CI, activation `x@V`, and pad-masked max CI at
the answer position with the batch axis kept (vs hand-rolled readouts); row-major `(a, b)`
reshape; the two-phase selection (host selection off max CI, only the shown columns
gathered); the shared-ordering prefix property across thresholds; n_alive / n_dropped
scalars; and the renderer emits valid CI + activation PNGs over the shared active set.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from param_decomp.arithmetic_eval import (
    ArithmeticGrid,
    ArithmeticSelection,
    ComponentActivationModel,
    compute_arithmetic_selection,
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


def test_grid_step_ci_xv_and_masked_max_match_hand_rolled():
    cfg, lm, ci_fn, C = _tiny_setup()
    assert isinstance(lm, ComponentActivationModel)
    vu = init_decomp_vu(lm.sites, jax.random.PRNGKey(1))
    n_pad = N_A * N_B + 2  # two garbage tail rows, as the sharding pad would append
    tokens = jax.random.randint(jax.random.PRNGKey(4), (n_pad, T), 0, cfg.vocab_size)
    step = make_arithmetic_grid_step(lm, ANSWER_POSITION, n_valid_rows=N_A * N_B)
    ci_grids, xv_grids, max_ci = step(lm, vu, ci_fn, tokens)

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
        ci = np.asarray(ci_grids[site])
        ci_exp = np.asarray(lower_leaky_hard_sigmoid(logits[site]))[:, ANSWER_POSITION, :]
        assert ci.shape == (n_pad, C)
        assert ci.min() >= 0.0 and ci.max() <= 1.0
        np.testing.assert_allclose(ci, ci_exp, rtol=1e-4, atol=1e-4)
        xv = np.asarray(xv_grids[site])
        assert xv.shape == (n_pad, C) and np.all(np.isfinite(xv))
        _, u = vu.vu[site]
        out_got = np.asarray(outputs[site])[:, ANSWER_POSITION, :].astype(np.float32)
        np.testing.assert_allclose(out_got, xv @ np.asarray(u, np.float32), atol=1e-2)
        # max CI is over the REAL rows only — the garbage tail must not decide liveness
        np.testing.assert_allclose(np.asarray(max_ci[site]), ci[: N_A * N_B].max(axis=0), rtol=1e-6)


def test_compute_arithmetic_selection_gathers_only_shown_columns():
    cfg, lm, ci_fn, _ = _tiny_setup()
    assert isinstance(lm, ComponentActivationModel)
    vu = init_decomp_vu(lm.sites, jax.random.PRNGKey(1))
    tokens = jax.random.randint(jax.random.PRNGKey(4), (N_A * N_B, T), 0, cfg.vocab_size)
    step = make_arithmetic_grid_step(lm, ANSWER_POSITION, n_valid_rows=N_A * N_B)
    top_k = 3
    selection = compute_arithmetic_selection(
        step, lm, vu, ci_fn, tokens, N_A * N_B, thresholds=(0.0,), top_k=top_k
    )
    full_ci, full_xv, _ = step(lm, vu, ci_fn, tokens)
    for site in lm.site_names:
        shown = selection.shown[site]
        assert shown.size == min(top_k, selection.active[0.0][site].size)
        assert selection.ci_columns[site].shape == (N_A * N_B, shown.size)
        np.testing.assert_allclose(
            selection.ci_columns[site], np.asarray(full_ci[site])[:, shown], rtol=1e-6
        )
        np.testing.assert_allclose(
            selection.xv_columns[site], np.asarray(full_xv[site])[:, shown], rtol=1e-6
        )


def test_to_grid_is_row_major_a_then_b():
    grid = _grid()
    per_prompt = np.array([[i * 10 + j] for i in range(N_A) for j in range(N_B)], dtype=float)
    reshaped = grid.to_grid(per_prompt)
    for i in range(N_A):
        for j in range(N_B):
            assert reshaped[i, j, 0] == i * 10 + j


def test_grid_rejects_non_contiguous_axes():
    with pytest.raises(AssertionError):  # the heatmap extent math assumes unit spacing
        ArithmeticGrid(a_values=(1, 3, 5), b_values=(1, 2), symbol="+")


def test_select_active_orders_by_max_and_is_prefix_across_thresholds():
    max_ci = {SITE: np.array([0.05, 0.9, 0.3, 0.95])}
    active = select_active(max_ci, (0.1, 0.5))
    assert active[0.1][SITE].tolist() == [3, 1, 2]  # 0 below thr; rest desc by max
    # a higher threshold's active set is a PREFIX of a lower's (one shared ordering)
    assert active[0.5][SITE].tolist() == [3, 1]
    assert select_active({SITE: np.full(5, 0.05)}, (0.1,))[0.1][SITE].size == 0


def test_select_active_and_n_alive_scalars():
    max_ci = {
        SITE: np.array([0.9, 0.2, 0.0]),  # 2 alive
        "layers.4.mlp.up_proj": np.array([0.05, 0.05]),  # 0 alive
    }
    active = select_active(max_ci, (0.1,))
    assert active[0.1][SITE].tolist() == [0, 1]
    assert active[0.1]["layers.4.mlp.up_proj"].size == 0
    scalars = n_alive_scalars(active, top_k=64)
    assert scalars["n_alive/thr0.1/" + SITE] == 2.0
    assert scalars["n_alive/thr0.1/layers.4.mlp.up_proj"] == 0.0
    assert scalars["n_alive/thr0.1/total"] == 2.0


def test_n_alive_scalars_reports_dropped_beyond_top_k():
    max_ci = np.array([0.8, 0.8, 0.8, 0.0, 0.0])  # 3 alive
    scalars = n_alive_scalars(select_active({SITE: max_ci}, (0.1,)), top_k=2)
    assert scalars["n_alive/thr0.1/" + SITE] == 3.0
    assert scalars["n_dropped/thr0.1/" + SITE] == 1.0  # 3 alive - top_k 2


def test_plot_component_grids_png_and_rejects_empty():
    grid = _grid()
    columns = np.zeros((N_A * N_B, 1))
    columns[:, 0] = 0.8
    png = plot_component_grids(columns, np.array([1]), grid, "t", "viridis", (0.0, 1.0))
    assert png[:4] == b"\x89PNG"
    signed = np.linspace(-2, 2, N_A * N_B * 2).reshape(N_A * N_B, 2)  # auto-symmetric range
    assert (
        plot_component_grids(signed, np.array([0, 1]), grid, "t", "coolwarm", None)[:4]
        == b"\x89PNG"
    )
    with pytest.raises(AssertionError):  # empty selection is a caller bug, not a silent no-op
        plot_component_grids(
            np.zeros((N_A * N_B, 0)), np.array([], dtype=int), grid, "t", "viridis", (0.0, 1.0)
        )


def test_render_arithmetic_figures_pairs_ci_and_activation():
    grid = _grid()
    max_ci = np.array([0.8, 0.7, 0.6, 0.0, 0.0])  # 3 alive, ids 0 > 1 > 2
    active = select_active({SITE: max_ci}, (0.1, 0.65))
    top_k = 2
    shown = {SITE: active[0.1][SITE][:top_k]}
    rng = np.random.default_rng(0)
    selection = ArithmeticSelection(
        active=active,
        shown=shown,
        ci_columns={SITE: rng.uniform(0, 1, (N_A * N_B, 2))},
        xv_columns={SITE: rng.standard_normal((N_A * N_B, 2))},
    )
    figures = render_arithmetic_figures(selection, grid, top_k)
    assert set(figures) == {
        f"figures/ci_grid/thr0.1/{SITE}",
        f"figures/activation_grid/thr0.1/{SITE}",
        f"figures/ci_grid/thr0.65/{SITE}",  # 2 alive at 0.65 -> a prefix of the shown columns
        f"figures/activation_grid/thr0.65/{SITE}",
    }
    for png in figures.values():
        assert png[:4] == b"\x89PNG"
