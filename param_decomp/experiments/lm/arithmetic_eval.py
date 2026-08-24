"""In-loop eval: per-component CI and activation (`x@V`) heatmaps over an `a x b` arithmetic
operand grid (the in-memory probe the lab builds in `experiments/lm/arithmetic_probe.py`).
See `param_decomp/CLAUDE.md`.
"""

import io
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from jaxtyping import Array, Float, Int
from matplotlib.figure import Figure
from typing_extensions import TypeVar

from param_decomp.core.ci_fn import PlacedCIFn, ci_preactivations, lower_leaky_hard_sigmoid
from param_decomp.core.components import ComponentStacks
from param_decomp.core.masking import all_live_masking_no_delta
from param_decomp.core.model import (
    CaptureKeys,
    DecomposedModel,
    MaterializedMasking,
    PlacedModel,
    prepare_compute_weights,
)
from param_decomp.core.placement import PlacementRules
from param_decomp.core.precision import COMPUTE_DT

PreparedT = TypeVar("PreparedT", default=Any)


@runtime_checkable
class ComponentActivationModel(DecomposedModel[PreparedT], Protocol[PreparedT]):
    """A `DecomposedModel` that also exposes per-component activations `x@V`. The arithmetic
    activation heatmaps need this seam; it is LM-only (currently `GLUDecomposedModel`), so
    the eval narrows to it with an `isinstance` check rather than widening the core
    `DecomposedModel` Protocol every target must satisfy."""

    def masked_component_activations(
        self,
        prepared_weights: PreparedT,
        inputs: Any,
        masking: MaterializedMasking,
        *,
        placement: PlacementRules | None,
    ) -> dict[str, Array]: ...


@dataclass(frozen=True)
class ArithmeticGrid:
    """The probe's grid geometry: row-major `(a, b)` order, `n_a * n_b == n_prompts`.
    Both axes are contiguous ascending integer ranges (asserted) — the heatmap extent
    math (integer cell centers, edges at ±0.5) relies on unit spacing."""

    a_values: tuple[int, ...]
    b_values: tuple[int, ...]
    symbol: str

    def __post_init__(self) -> None:
        for values in (self.a_values, self.b_values):
            assert values == tuple(range(values[0], values[-1] + 1)), values

    @property
    def n_a(self) -> int:
        return len(self.a_values)

    @property
    def n_b(self) -> int:
        return len(self.b_values)

    def to_grid(self, per_prompt: np.ndarray) -> np.ndarray:
        """Reshape a `(n_prompts, ...)` array (row-major `(a, b)`) to `(n_a, n_b, ...)`."""
        assert per_prompt.shape[0] == self.n_a * self.n_b, (per_prompt.shape, self.n_a, self.n_b)
        return per_prompt.reshape(self.n_a, self.n_b, *per_prompt.shape[1:])


ArithmeticGridStep = Callable[
    [PlacedModel, ComponentStacks, PlacedCIFn, Int[Array, "n_pad T"]],
    tuple[dict[str, Array], dict[str, Array], dict[str, Array]],
]
"""`(model, components, placed_ci_fn, tokens) -> ({site: CI}, {site: x@V}, {site: max CI})`. CI and
`x@V` are `(n_pad, C)` at the answer position, batch-sharded ON DEVICE (never fully
host-gathered — see `compute_arithmetic_selection`); max CI is `(C,)` replicated, the max
over the REAL (`< n_valid_rows`) rows only. `model` (frozen-weight-bearing) is the jit ARG."""


def component_activation_model[PreparedT](
    placed: PlacedModel[PreparedT],
) -> ComponentActivationModel[PreparedT]:
    """Narrow the bundle's target to the `x@V` seam — re-derived from the traced model arg
    each call, never closed over (the HLO-baking rule)."""
    model = placed.model
    assert isinstance(model, ComponentActivationModel), (
        f"arithmetic eval needs masked_component_activations; {type(model).__name__} does not"
    )
    return model


def make_arithmetic_grid_step[PreparedT](
    model_static: PlacedModel[PreparedT],
    ci_capture_keys: CaptureKeys,
    answer_position: int,
    n_valid_rows: int,
) -> ArithmeticGridStep:
    """Build the jit'd step returning, at `answer_position` with the batch axis KEPT as the
    grid, BOTH per-component lower-leaky CI (from the CI fn) and the pre-mask activation `x@V`
    (from the decomposed forward under all-ones masks ≈ full reconstruction), plus the
    per-component max CI over the real rows (`n_valid_rows` masks the sharding-pad tail —
    garbage prompts must not decide liveness). One step so the grids come from one call; a
    site's own mask never enters its `x@V`."""
    component_activation_model(model_static)
    site_names = model_static.site_names

    # HLO-baking rule: read STATIC config (site_names, Cs) off the closed-over `model_static`; all array
    # access goes through the traced `model` arg.
    @eqx.filter_jit
    def step(
        model: PlacedModel[PreparedT],
        components: ComponentStacks,
        placed_ci_fn: PlacedCIFn,
        tokens: Int[Array, "n_pad T"],
    ) -> tuple[dict[str, Array], dict[str, Array], dict[str, Array]]:
        preactivations = ci_preactivations(
            placed_ci_fn,
            model.clean_forward(tokens, ci_capture_keys).captures,
            remat=False,
        )
        assert preactivations[site_names[0]].ndim == 3, (
            f"arithmetic grid is LM-only ((n_prompts, T, C)); got {preactivations[site_names[0]].shape}"
        )
        answer_position_ci = {
            site: lower_leaky_hard_sigmoid(preactivations[site])[:, answer_position, :]
            for site in site_names
        }

        prepared_weights = prepare_compute_weights(model, components)
        component_activations = component_activation_model(model).masked_component_activations(
            prepared_weights,
            tokens,
            all_live_masking_no_delta(
                model_static.sites, leading_shape=tokens.shape, dtype=COMPUTE_DT
            ),
            placement=model.placement,
        )
        answer_position_component_activations = {
            site: component_activations[site][:, answer_position, :].astype(jnp.float32)
            for site in site_names
        }
        valid_rows = (jnp.arange(tokens.shape[0]) < n_valid_rows)[:, None]
        max_ci_by_component = {
            site: jnp.where(valid_rows, answer_position_ci[site], -jnp.inf).max(axis=0)
            for site in site_names
        }
        return (
            answer_position_ci,
            answer_position_component_activations,
            max_ci_by_component,
        )

    return step


@eqx.filter_jit
def _take_columns(per_prompt: Float[Array, "n_pad C"], idx: Int[Array, " k"]) -> Array:
    return jnp.take(per_prompt, idx, axis=1)


@dataclass(frozen=True)
class ArithmeticSelection:
    """Host-side result of one arithmetic grid pass: the per-threshold active sets and the
    gathered per-prompt columns for the (at most `top_k`) plotted components. All of a
    site's index arrays share ONE descending-max-CI ordering, so every threshold's plotted
    set is a PREFIX of that site's `shown` (a higher threshold keeps fewer components of
    the same ordering) — the figures index columns by prefix, never by search."""

    active: dict[float, dict[str, np.ndarray]]
    """`{threshold: {site: component ids with max CI > threshold, desc by max CI}}`."""
    shown: dict[str, np.ndarray]
    """`{site: the component ids whose columns were gathered}` — the first `top_k` active
    at the lowest threshold (the superset of every threshold's plotted set)."""
    ci_columns: dict[str, np.ndarray]
    """`{site: (n_prompts, len(shown[site]))}` fp32, row-major `(a, b)` order."""
    xv_columns: dict[str, np.ndarray]


def compute_arithmetic_selection(
    step: ArithmeticGridStep,
    model: PlacedModel,
    components: ComponentStacks,
    placed_ci_fn: PlacedCIFn,
    tokens: Int[Array, "n_pad T"],
    n_prompts: int,
    thresholds: tuple[float, ...],
    top_k: int,
) -> ArithmeticSelection:
    """Two-phase device->host pull, sized to what the figures actually need — NEVER the full
    `(n_prompts, C)` grids (they scale as n_prompts x C per site). Phase 1: the step's replicated
    per-component max CI comes to host (a `(C,)` vector per site) and drives the selection,
    identically on every rank. Phase 2: only the selected columns are host-gathered (the
    index padded to `top_k` so the gather compiles once; `n_prompts` trims the sharding pad
    rows off the END). Both the step and the column gather are COLLECTIVE — all ranks join."""
    ci, xv, max_ci_replicated = step(model, components, placed_ci_fn, tokens)
    max_ci = {s: np.asarray(v) for s, v in max_ci_replicated.items()}
    active = select_active(max_ci, thresholds)
    shown = {s: active[min(thresholds)][s][:top_k] for s in max_ci}

    ci_columns: dict[str, np.ndarray] = {}
    xv_columns: dict[str, np.ndarray] = {}
    to_gather: dict[str, tuple[Array, Array]] = {}
    for site, idx in shown.items():
        if idx.size == 0:
            ci_columns[site] = np.zeros((n_prompts, 0), np.float32)
            xv_columns[site] = np.zeros((n_prompts, 0), np.float32)
            continue
        padded_idx = jnp.asarray(np.pad(idx, (0, top_k - idx.size), mode="edge"))
        to_gather[site] = (_take_columns(ci[site], padded_idx), _take_columns(xv[site], padded_idx))
    if to_gather:
        gathered = multihost_utils.process_allgather(to_gather, tiled=True)
        for site, (ci_cols, xv_cols) in gathered.items():
            k = shown[site].size
            ci_columns[site] = np.asarray(ci_cols)[:n_prompts, :k]
            xv_columns[site] = np.asarray(xv_cols)[:n_prompts, :k]
    return ArithmeticSelection(
        active=active, shown=shown, ci_columns=ci_columns, xv_columns=xv_columns
    )


def select_active(
    max_ci: dict[str, np.ndarray], thresholds: tuple[float, ...]
) -> dict[float, dict[str, np.ndarray]]:
    """`{threshold: {site: ids of components whose max CI over the grid exceeds the
    threshold}}`, descending by max CI so the most-active plot first. Computed ONCE so the
    `n_alive` scalars and the figures share the same selection, with ONE stable ordering
    per site across all thresholds — a higher threshold's active set is a prefix of a
    lower's (`ArithmeticSelection` relies on this)."""
    out: dict[float, dict[str, np.ndarray]] = {t: {} for t in thresholds}
    for site, site_max_ci in max_ci.items():
        assert site_max_ci.ndim == 1, site_max_ci.shape
        order = np.argsort(-site_max_ci, kind="stable")
        for t in thresholds:
            out[t][site] = order[site_max_ci[order] > t]
    return out


def n_alive_scalars(active: dict[float, dict[str, np.ndarray]], top_k: int) -> dict[str, float]:
    """Un-prefixed scalars from the active sets: `n_alive/thr<t>/<site>` (+ `/total`) counts ALL
    alive components (high at init, falls as training sparsifies), and
    `n_dropped/thr<t>/<site>` reports any alive beyond the `top_k` plotted (no silent cap)."""
    out: dict[str, float] = {}
    for t, per_site in active.items():
        total = 0.0
        for site, idx in per_site.items():
            out[f"n_alive/thr{t:g}/{site}"] = float(idx.size)
            if idx.size > top_k:
                out[f"n_dropped/thr{t:g}/{site}"] = float(idx.size - top_k)
            total += float(idx.size)
        out[f"n_alive/thr{t:g}/total"] = total
    return out


def _render_figure(fig: Figure) -> bytes:
    """Encode a standalone `Figure` to PNG bytes. Object-oriented `Figure`, never `pyplot`:
    this tier renders on `BackgroundRenderer`'s worker thread, where pyplot's global figure
    registry is unsynchronized and an interactive backend cannot build a figure manager at
    all (see `slow_eval._render_figure`)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    return buf.getvalue()


def plot_component_grids(
    columns: np.ndarray,
    component_ids: np.ndarray,
    grid: ArithmeticGrid,
    title: str,
    cmap: str,
    value_range: tuple[float, float] | None,
) -> bytes:
    """One faceted figure: an `a x b` heatmap per column of `columns` (`(n_prompts, k)`,
    non-empty), panel `j` titled by `component_ids[j]`. Cell `(i, j)` of a panel is the
    component's value for `a_values[i] <op> b_values[j] =`; the imshow extent is the outer
    CELL EDGES (integer centers ± 0.5, unit spacing pinned by `ArithmeticGrid`) so ticks
    land on operand values. `value_range` fixes the color scale (e.g. (0, 1) for CI); None
    auto-scales SYMMETRICALLY about 0 (signed activations)."""
    n = component_ids.size
    assert columns.ndim == 2 and columns.shape[1] == n > 0, (columns.shape, n)
    grids = grid.to_grid(columns)
    if value_range is None:
        vmax = float(np.abs(grids).max())
        vmin, vmax = (-vmax, vmax) if vmax > 0 else (-1.0, 1.0)
    else:
        vmin, vmax = value_range
    n_cols = min(n, 8)
    n_rows = (n + n_cols - 1) // n_cols
    fig = Figure(figsize=(2.4 * n_cols, 2.6 * n_rows), layout="constrained")
    axs = fig.subplots(n_rows, n_cols, squeeze=False)
    flat = axs.ravel()
    for ax in flat[n:]:
        ax.set_visible(False)
    images = []
    for j, (ax, component_id) in enumerate(zip(flat, component_ids, strict=False)):
        images.append(
            ax.imshow(
                grids[:, :, j],
                aspect="auto",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                origin="lower",
                extent=(
                    grid.b_values[0] - 0.5,
                    grid.b_values[-1] + 0.5,
                    grid.a_values[0] - 0.5,
                    grid.a_values[-1] + 0.5,
                ),
            )
        )
        ax.set_title(f"c{int(component_id)}", fontsize=8)
        ax.tick_params(labelsize=6)
    fig.colorbar(images[0], ax=axs.ravel().tolist(), shrink=0.6)
    fig.suptitle(title)
    return _render_figure(fig)


def render_arithmetic_figures(
    selection: ArithmeticSelection, grid: ArithmeticGrid, top_k: int
) -> dict[str, bytes]:
    """Per `(threshold, site)` with any active component: a CI heatmap and the matching `x@V`
    activation heatmap over the top-`top_k` active components. Each threshold's plotted set
    is a PREFIX of the site's gathered `shown` columns (asserted; see `ArithmeticSelection`).
    Keyed `figures/{ci_grid,activation_grid}/thr<t>/<site>`."""
    figures: dict[str, bytes] = {}
    for t, per_site in selection.active.items():
        for site, idx in per_site.items():
            n_shown = min(top_k, idx.size)
            if n_shown == 0:
                continue
            shown = selection.shown[site][:n_shown]
            assert np.array_equal(idx[:n_shown], shown), (t, site)
            figures[f"figures/ci_grid/thr{t:g}/{site}"] = plot_component_grids(
                selection.ci_columns[site][:, :n_shown],
                shown,
                grid,
                f"{site} CI ({grid.symbol} grid, thr={t:g})",
                cmap="viridis",
                value_range=(0.0, 1.0),
            )
            figures[f"figures/activation_grid/thr{t:g}/{site}"] = plot_component_grids(
                selection.xv_columns[site][:, :n_shown],
                shown,
                grid,
                f"{site} activation x@V ({grid.symbol} grid, thr={t:g})",
                cmap="coolwarm",
                value_range=None,
            )
    return figures
