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
from jaxtyping import Array, Int
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from param_decomp.ci_fn import lower_leaky_hard_sigmoid
from param_decomp.components import DecompVU
from param_decomp.lm import DecomposedModel
from param_decomp.train import COMPUTE_DT, cast_floating


@runtime_checkable
class ComponentActivationModel(DecomposedModel, Protocol):
    """A `DecomposedModel` that also exposes per-component activations `x@V`. The arithmetic
    activation heatmaps need this seam; it is LM-only (currently `LlamaDecomposedModel`), so
    the eval narrows to it with an `isinstance` check rather than widening the core
    `DecomposedModel` Protocol every target must satisfy."""

    def masked_component_activations(
        self,
        prepared: Any,
        inputs: Any,
        masks: dict[str, Array],
        delta_masks: dict[str, Array],
        routes: dict[str, Array] | None,
        live: tuple[str, ...],
        has_delta: bool,
    ) -> dict[str, Array]: ...


@dataclass(frozen=True)
class ArithmeticGrid:
    """The probe's grid geometry: row-major `(a, b)` order, `n_a * n_b == n_prompts`."""

    a_values: tuple[int, ...]
    b_values: tuple[int, ...]
    symbol: str

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
    [ComponentActivationModel, DecompVU, Any, Int[Array, "n_prompts T"]],
    tuple[dict[str, Array], dict[str, Array]],
]
"""`(model, components, ci_fn, tokens) -> ({site: CI}, {site: x@V})`, each `(n_prompts, C)`
at the answer position. `model` (frozen-weight-bearing) is the jit ARG."""


def make_arithmetic_grid_step(
    lm: ComponentActivationModel, answer_position: int
) -> ArithmeticGridStep:
    """Build the jit'd step returning, at `answer_position` with the batch axis KEPT as the
    grid, BOTH per-component lower-leaky CI (from the CI fn) and the pre-mask activation `x@V`
    (from the decomposed forward under all-ones masks ≈ full reconstruction). One step so the
    two grids come from one call; a site's own mask never enters its `x@V`."""
    site_names = lm.site_names
    site_component_counts = {s.name: s.C for s in lm.sites}

    # HLO-baking rule: read STATIC config (site_names, Cs) off the closed-over `lm`; all array
    # access goes through the traced `model` arg.
    @eqx.filter_jit
    def step(
        model: ComponentActivationModel,
        components: DecompVU,
        ci_fn: Any,
        tokens: Int[Array, "n_prompts T"],
    ) -> tuple[dict[str, Array], dict[str, Array]]:
        ci_fn = cast_floating(ci_fn, COMPUTE_DT)
        taps = {
            k: x.astype(COMPUTE_DT)
            for k, x in model.read_activations(tokens, ci_fn.input_names).items()
        }
        logits = {s: v.astype(jnp.float32) for s, v in ci_fn(taps, remat=False).logits.items()}
        assert logits[site_names[0]].ndim == 3, (
            f"arithmetic grid is LM-only ((n_prompts, T, C)); got {logits[site_names[0]].shape}"
        )
        ci = {s: lower_leaky_hard_sigmoid(logits[s])[:, answer_position, :] for s in site_names}

        prepared = model.prepare_compute_weights(cast_floating(components, COMPUTE_DT))
        leading = tokens.shape
        ones = {s: jnp.ones((*leading, site_component_counts[s]), COMPUTE_DT) for s in site_names}
        zeros_delta = {s: jnp.zeros(leading, COMPUTE_DT) for s in site_names}
        acts = model.masked_component_activations(
            prepared, tokens, ones, zeros_delta, None, site_names, False
        )
        xv = {s: acts[s][:, answer_position, :].astype(jnp.float32) for s in site_names}
        return ci, xv

    return step


def accumulate_arithmetic_grids(
    step: ArithmeticGridStep,
    model: ComponentActivationModel,
    components: DecompVU,
    ci_fn: Any,
    token_batches: list[Int[Array, "n_prompts T"]],
    n_prompts: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Run `step` over the probe batches and host-gather to `(ci_grids, xv_grids)`, each
    `{site: (n_prompts, C)}` in row-major `(a, b)` order. The probe is small (typically one
    batch), but a padded grid may be split; `n_prompts` trims the sharding pad rows off the
    END (pad appends after the real grid). The gather is COLLECTIVE — all ranks join it."""
    assert token_batches, "arithmetic grid needs at least one batch"
    ci_parts: dict[str, list[np.ndarray]] = {}
    xv_parts: dict[str, list[np.ndarray]] = {}
    for tokens in token_batches:
        ci, xv = step(model, components, ci_fn, tokens)
        for parts, per_site in ((ci_parts, ci), (xv_parts, xv)):
            for site, vals in per_site.items():
                parts.setdefault(site, []).append(
                    np.asarray(multihost_utils.process_allgather(vals, tiled=True))
                )
    ci_grids = {s: np.concatenate(p)[:n_prompts] for s, p in ci_parts.items()}
    xv_grids = {s: np.concatenate(p)[:n_prompts] for s, p in xv_parts.items()}
    return ci_grids, xv_grids


def active_components(ci_per_prompt: np.ndarray, threshold: float) -> np.ndarray:
    """Indices of components whose MAX CI over the grid exceeds `threshold` (alive on this
    probe), ordered by descending max CI so the most-active plot first. `ci_per_prompt` is
    `(n_prompts, C)`."""
    assert ci_per_prompt.ndim == 2, ci_per_prompt.shape
    max_ci = ci_per_prompt.max(axis=0)
    alive = np.flatnonzero(max_ci > threshold)
    return alive[np.argsort(-max_ci[alive])]


def select_active(
    ci_grids: dict[str, np.ndarray], thresholds: tuple[float, ...]
) -> dict[float, dict[str, np.ndarray]]:
    """`{threshold: {site: active component indices}}`, computed ONCE so the `n_alive` scalars
    and the figures share the same selection."""
    return {t: {s: active_components(ci, t) for s, ci in ci_grids.items()} for t in thresholds}


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
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def plot_component_grids(
    per_prompt: np.ndarray,
    grid: ArithmeticGrid,
    component_indices: np.ndarray,
    title: str,
    cmap: str,
    value_range: tuple[float, float] | None,
) -> bytes:
    """One faceted figure: an `a x b` heatmap per component in `component_indices` (already
    top-k-capped, non-empty). `per_prompt` is `(n_prompts, C)`; cell `(i, j)` of panel `c` is
    component `c`'s value for `a_values[i] <op> b_values[j] =`. `value_range` fixes the color
    scale (e.g. (0, 1) for CI); None auto-scales SYMMETRICALLY about 0 (signed activations)."""
    assert component_indices.size > 0, "plot_component_grids needs at least one component"
    grids = grid.to_grid(per_prompt)
    if value_range is None:
        vmax = float(np.abs(grids[:, :, component_indices]).max())
        vmin, vmax = (-vmax, vmax) if vmax > 0 else (-1.0, 1.0)
    else:
        vmin, vmax = value_range
    n = component_indices.size
    n_cols = min(n, 8)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.4 * n_cols, 2.6 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )
    flat = axs.ravel()
    for ax in flat[n:]:
        ax.set_visible(False)
    images = []
    for ax, c in zip(flat, component_indices, strict=False):
        images.append(
            ax.imshow(
                grids[:, :, c],
                aspect="auto",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                origin="lower",
                extent=(grid.b_values[0], grid.b_values[-1], grid.a_values[0], grid.a_values[-1]),
            )
        )
        ax.set_title(f"c{int(c)}", fontsize=8)
        ax.tick_params(labelsize=6)
    fig.colorbar(images[0], ax=axs.ravel().tolist(), shrink=0.6)
    fig.suptitle(title)
    return _render_figure(fig)


def render_arithmetic_figures(
    ci_grids: dict[str, np.ndarray],
    xv_grids: dict[str, np.ndarray],
    active: dict[float, dict[str, np.ndarray]],
    grid: ArithmeticGrid,
    top_k: int,
) -> dict[str, bytes]:
    """Per `(threshold, site)` with any active component: a CI heatmap and the matching `x@V`
    activation heatmap over the SAME top-`top_k` active components (from the precomputed
    `active` selection). Keyed `figures/{ci_grid,activation_grid}/thr<t>/<site>`."""
    figures: dict[str, bytes] = {}
    for t, per_site in active.items():
        for site, idx in per_site.items():
            shown = idx[:top_k]
            if shown.size == 0:
                continue
            figures[f"figures/ci_grid/thr{t:g}/{site}"] = plot_component_grids(
                ci_grids[site],
                grid,
                shown,
                f"{site} CI ({grid.symbol} grid, thr={t:g})",
                cmap="viridis",
                value_range=(0.0, 1.0),
            )
            figures[f"figures/activation_grid/thr{t:g}/{site}"] = plot_component_grids(
                xv_grids[site],
                grid,
                shown,
                f"{site} activation x@V ({grid.symbol} grid, thr={t:g})",
                cmap="coolwarm",
                value_range=None,
            )
    return figures
