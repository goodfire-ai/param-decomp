"""Interactive Panel/Bokeh app for browsing the hierarchically clustered coactivation matrix.

Loads the `.npz` written by `plot_initial_coactivation.py` and serves an interactive view:

  - Bokeh `image_url` heatmap of `log10 P(i ∧ j)` — the matrix is pre-rendered server-side
    to a PNG and embedded as a base64 data URL, so the displayed image is full 1:1
    resolution (one source-component per pixel) without paying for a 380 MB float32 push.
  - JS slider for the dendrogram cut (`n_clusters`) — pre-computed cuts indexed by a
    discrete slider, with a `CustomJS` callback updating both the cluster strip glyph
    and the slider title.
  - Two `Rect` strips (one rect per component) above the heatmap: module identity (static)
    and cluster id at the current cut (updates with the slider).
  - JS hover (`MouseMove` event) updates an info Div with row/col indices, short and full
    canonical labels, and cluster ids at the current cut.

All interactivity is `CustomJS`, so the same code path works for the served Panel app and
for the self-contained HTML snapshot.

Usage (on the dev node):
    cd ~/spd && source .venv/bin/activate
    python -m spd.clustering.scripts.interactive_coactivation \
        ~/spd/spd/clustering/scripts/out/ch-2b9414a4/coactivation_clustered.npz \
        --port 5179 --address 0.0.0.0

    # Or save a self-contained HTML snapshot:
    python -m spd.clustering.scripts.interactive_coactivation \
        <npz_path> --save_html /tmp/coact_standalone.html
"""

import base64
import io
import logging
from pathlib import Path

import fire
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from bokeh.events import MouseMove
from bokeh.layouts import column, row
from bokeh.models import (
    BasicTicker,
    Button,
    CheckboxGroup,
    ColorBar,
    ColumnDataSource,
    CustomJS,
    Div,
    LinearColorMapper,
    Slider,
)
from bokeh.palettes import Category20, Turbo256, Viridis256
from bokeh.plotting import figure
from jinja2 import Template
from numpy.typing import NDArray
from PIL import Image
from scipy.cluster.hierarchy import fcluster
from spd.log import logger

LOADING_TEMPLATE = Template("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{{ title if title else "SPD coactivation viewer" }}</title>
{{ bokeh_css }}
{{ bokeh_js }}
<style>
#spd-loading {
    position: fixed; top: 0; left: 0; right: 0; bottom: 0; z-index: 999999;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    background: rgba(255,255,255,0.97);
    font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif;
    gap: 14px;
    color: #333;
}
.spd-spinner {
    width: 56px; height: 56px;
    border: 5px solid #e0e0e0; border-top-color: #2b7cd3;
    border-radius: 50%;
    animation: spd-spin 1s linear infinite;
}
@keyframes spd-spin { to { transform: rotate(360deg); } }
#spd-loading .spd-msg { font-size: 14px; font-weight: 500; }
#spd-loading .spd-sub { font-size: 12px; color: #888; }
</style>
</head>
<body>
<div id="spd-loading">
    <div class="spd-spinner"></div>
    <div class="spd-msg">Loading coactivation viewer…</div>
    <div class="spd-sub">~30 MB transfer; usually 5&ndash;30 seconds over the tunnel.</div>
</div>
{{ plot_div|safe }}
{{ plot_script|safe }}
<script>
(function() {
    function hide() {
        const el = document.getElementById("spd-loading");
        if (el) el.remove();
    }
    function waitForBokeh() {
        if (window.Bokeh && Bokeh.documents && Bokeh.documents.length > 0) {
            const doc = Bokeh.documents[0];
            if (doc.is_idle) { hide(); return; }
            doc.idle.connect(hide);
            return;
        }
        setTimeout(waitForBokeh, 100);
    }
    waitForBokeh();
    setTimeout(hide, 120000);
})();
</script>
</body>
</html>
""")

logging.getLogger("bokeh").setLevel(logging.INFO)
logging.getLogger("tornado.access").setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")

MODULE_SHORTCODE = {
    "attn.q_proj": "q",
    "attn.k_proj": "k",
    "attn.v_proj": "v",
    "attn.o_proj": "o",
    "mlp.gate_proj": "g",
    "mlp.up_proj": "u",
    "mlp.down_proj": "d",
}


def _short_label(full_label: str) -> str:
    layer_module, _, cidx = full_label.rpartition(":")
    if not layer_module.startswith("h."):
        return full_label
    parts = layer_module.split(".", 2)
    if len(parts) < 3:
        return full_label
    layer = parts[1]
    rest = parts[2]
    short = MODULE_SHORTCODE.get(rest, rest)
    return f"{layer}{short}:{cidx}"


def _categorical_palette(n_categories: int) -> list[str]:
    if n_categories <= 20:
        return list(Category20[max(3, min(20, n_categories))])[:n_categories]
    base = list(Turbo256)
    if n_categories >= len(base):
        return [base[(i * 257) % len(base)] for i in range(n_categories)]
    step = len(base) / n_categories
    return [base[int(i * step)] for i in range(n_categories)]


def _render_heatmap_image(
    log_p_both: NDArray[np.float32],
    log_low: float,
    log_high: float,
    max_pixels: int,
    quality: int,
) -> tuple[bytes, str, int]:
    """Render the heatmap to a compact image (JPEG).

    Returns (bytes, mime_type, rendered_dim). `max_pixels` caps either dimension; we
    block-mean the matrix when n > max_pixels so the source array we hand to PIL stays
    manageable. JPEG is fine because exact pixel values are not used for hover — labels
    and cluster ids come from the original arrays.
    """
    n = log_p_both.shape[0]
    if n > max_pixels:
        factor = (n + max_pixels - 1) // max_pixels
        h2 = (n // factor) * factor
        log_p_both = (
            log_p_both[:h2, :h2]
            .reshape(h2 // factor, factor, h2 // factor, factor)
            .mean(axis=(1, 3))
        )
    rendered_n = log_p_both.shape[0]

    norm = mcolors.Normalize(vmin=log_low, vmax=log_high)
    mapper = cm.ScalarMappable(norm=norm, cmap="viridis")
    rgba = mapper.to_rgba(log_p_both, bytes=True)
    img = Image.fromarray(rgba[..., :3], mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True, progressive=False)
    return buf.getvalue(), "image/jpeg", rendered_n


class _Prepared:
    """Heavy data prepared once; cheap Bokeh models built per session."""

    def __init__(self, npz_path: Path, n_cluster_steps: int, heatmap_max_pixels: int) -> None:
        logger.info(f"loading {npz_path}")
        data = np.load(npz_path, allow_pickle=False)
        coact = data["coact"]
        order = data["order"].astype(np.int64)
        Z = data["linkage"].astype(np.float64)
        full_labels = [str(x) for x in data["labels"]]
        n_samples = int(data["n_samples"][0])

        n = len(order)
        logger.info(f"loaded k={n}, n_samples={n_samples:,}")

        coact_reordered = coact[order][:, order]
        p_both = coact_reordered.astype(np.float64) / n_samples
        log_p_both = np.log10(np.clip(p_both, 1.0 / n_samples, None)).astype(np.float32)
        del coact, coact_reordered, p_both
        log_low = float(np.nanmin(log_p_both))
        log_high = float(np.nanmax(log_p_both))

        img_bytes, mime_type, rendered_n = _render_heatmap_image(
            log_p_both, log_low, log_high, heatmap_max_pixels, quality=75
        )
        del log_p_both
        self.data_url = f"data:{mime_type};base64," + base64.b64encode(img_bytes).decode("ascii")
        self.rendered_n = rendered_n
        self.log_low = log_low
        self.log_high = log_high
        logger.info(
            f"rendered heatmap: {rendered_n}² pixels, {len(img_bytes) / 1e6:.2f} MB ({mime_type})"
        )

        self.full_labels_reordered = [full_labels[i] for i in order]
        self.short_labels_reordered = [_short_label(lab) for lab in self.full_labels_reordered]
        self.n = n
        self.npz_path = npz_path

        modules_reordered = [self.full_labels_reordered[i].rsplit(":", 1)[0] for i in range(n)]
        self.unique_modules = sorted(set(modules_reordered))
        module_to_id = {m: i for i, m in enumerate(self.unique_modules)}
        self.module_ids = np.array([module_to_id[m] for m in modules_reordered], dtype=np.int32)

        n_clusters_grid = np.unique(np.round(np.geomspace(2, n, num=n_cluster_steps)).astype(int))
        cluster_matrix = np.zeros((len(n_clusters_grid), n), dtype=np.int32)
        for idx, nc in enumerate(n_clusters_grid):
            cluster_matrix[idx] = fcluster(Z, t=int(nc), criterion="maxclust")
        self.cluster_matrix_reordered = cluster_matrix[:, order]
        self.n_clusters_grid = n_clusters_grid
        logger.info(
            f"precomputed cluster assignments at {len(n_clusters_grid)} cut levels: "
            f"{n_clusters_grid[0]}..{n_clusters_grid[-1]}"
        )


def _build_bokeh_layout(prepared: _Prepared) -> column:
    n = prepared.n
    data_url = prepared.data_url
    rendered_n = prepared.rendered_n
    log_low = prepared.log_low
    log_high = prepared.log_high
    full_labels_reordered = prepared.full_labels_reordered
    short_labels_reordered = prepared.short_labels_reordered
    unique_modules = prepared.unique_modules
    module_ids = prepared.module_ids
    n_clusters_grid = prepared.n_clusters_grid
    cluster_matrix_reordered = prepared.cluster_matrix_reordered

    HEAT_PIXELS = 820
    fig = figure(
        width=HEAT_PIXELS + 90,
        height=HEAT_PIXELS,
        x_range=(0, n),
        y_range=(n, 0),
        toolbar_location="right",
        tools="pan,box_zoom,wheel_zoom,reset,save",
        active_scroll="wheel_zoom",
        title=f"Initial coactivation log10 P(both) — k={n}, PNG rendered at {rendered_n}²",
    )
    fig.xaxis.axis_label = "component (reordered by hierarchical clustering)"
    fig.yaxis.axis_label = "component (reordered by hierarchical clustering)"
    fig.image_url(
        url=[data_url],
        x=0,
        y=0,
        w=n,
        h=n,
        anchor="top_left",
        level="image",
        global_alpha=1.0,
    )

    heat_mapper_for_colorbar = LinearColorMapper(
        palette=list(Viridis256), low=log_low, high=log_high
    )
    color_bar = ColorBar(
        color_mapper=heat_mapper_for_colorbar, ticker=BasicTicker(), title="log10 P(both)"
    )
    fig.add_layout(color_bar, "right")

    STRIP_HEIGHT = 24

    def _make_strip_figure() -> figure:
        strip = figure(
            width=HEAT_PIXELS + 90,
            height=STRIP_HEIGHT,
            x_range=fig.x_range,
            y_range=(0, 1),
            toolbar_location=None,
            tools="",
            min_border=0,
        )
        strip.xaxis.visible = False
        strip.yaxis.visible = False
        strip.grid.visible = False
        return strip

    module_palette = _categorical_palette(len(unique_modules))
    module_strip = _make_strip_figure()
    initial_module_colors = [module_palette[int(m)] for m in module_ids.tolist()]
    module_strip_source = ColumnDataSource(
        data=dict(
            x=(np.arange(n, dtype=np.float32) + 0.5).tolist(),
            y=[0.5] * n,
            module_id=module_ids.tolist(),
            fill_color=initial_module_colors,
        )
    )
    module_strip.rect(
        x="x",
        y="y",
        width=1.0,
        height=1.0,
        source=module_strip_source,
        fill_color="fill_color",
        line_color=None,
    )

    initial_cut_idx = int(min(20, len(n_clusters_grid) - 1))

    palette = list(Turbo256)
    palette_size = len(palette)
    rng = np.random.default_rng(seed=42)
    palette_perm = rng.permutation(palette_size)

    def _cluster_id_to_color(cid: int) -> str:
        return palette[int(palette_perm[cid % palette_size])]

    color_matrix = np.empty(cluster_matrix_reordered.shape, dtype=object)
    for r in range(cluster_matrix_reordered.shape[0]):
        for c in range(cluster_matrix_reordered.shape[1]):
            color_matrix[r, c] = _cluster_id_to_color(int(cluster_matrix_reordered[r, c]))
    initial_colors = color_matrix[initial_cut_idx].tolist()

    cluster_strip = _make_strip_figure()
    cluster_strip_source = ColumnDataSource(
        data=dict(
            x=(np.arange(n, dtype=np.float32) + 0.5).tolist(),
            y=[0.5] * n,
            fill_color=initial_colors,
        )
    )
    cluster_strip.rect(
        x="x",
        y="y",
        width=1.0,
        height=1.0,
        source=cluster_strip_source,
        fill_color="fill_color",
        line_color=None,
    )

    slider = Slider(
        start=0,
        end=len(n_clusters_grid) - 1,
        value=initial_cut_idx,
        step=1,
        title=f"cut index (n_clusters = {int(n_clusters_grid[initial_cut_idx])})",
        width=HEAT_PIXELS,
    )

    n_clusters_grid_js = [int(x) for x in n_clusters_grid]
    cluster_matrix_js = cluster_matrix_reordered.astype(np.int32).tolist()
    color_matrix_js = color_matrix.tolist()

    info_div = Div(
        text=(
            "<pre style='margin:0; font-family:monospace; font-size:12px'>"
            "Hover over the heatmap to see component / cluster info."
            "</pre>"
        ),
        width=HEAT_PIXELS,
        height=110,
        styles={"background": "#f5f5f5", "border": "1px solid #ddd", "padding": "8px"},
    )

    hover_callback = CustomJS(
        args=dict(
            div=info_div,
            full_labels=full_labels_reordered,
            short_labels=short_labels_reordered,
            cluster_matrix=cluster_matrix_js,
            n_clusters_grid=n_clusters_grid_js,
            slider=slider,
            n=n,
        ),
        code="""
        const x = cb_obj.x;
        const y = cb_obj.y;
        if (x === null || y === null) { return; }
        const j = Math.min(Math.max(Math.floor(x), 0), n - 1);
        const i = Math.min(Math.max(Math.floor(y), 0), n - 1);
        const cut_idx = slider.value;
        const nc = n_clusters_grid[cut_idx];
        const ci = cluster_matrix[cut_idx][i];
        const cj = cluster_matrix[cut_idx][j];
        const lab_i = full_labels[i];
        const lab_j = full_labels[j];
        const slab_i = short_labels[i];
        const slab_j = short_labels[j];
        div.text =
            "<pre style='margin:0; font-family:monospace; font-size:12px'>" +
            "row " + i.toString().padStart(5) + " " + slab_i.padEnd(10) +
              "  cluster " + ci.toString().padStart(5) +
              "  " + lab_i + "\\n" +
            "col " + j.toString().padStart(5) + " " + slab_j.padEnd(10) +
              "  cluster " + cj.toString().padStart(5) +
              "  " + lab_j + "\\n" +
            "(n_clusters = " + nc + ")" +
            "</pre>";
        """,
    )
    fig.js_on_event(MouseMove, hover_callback)

    overlay_xs = (np.arange(n, dtype=np.float32) + 0.5).tolist()
    col_overlay_source = ColumnDataSource(data=dict(x=overlay_xs, alpha=[0.0] * n))
    row_overlay_source = ColumnDataSource(data=dict(y=overlay_xs, alpha=[0.0] * n))
    fig.vbar(
        x="x",
        width=1.0,
        bottom=0,
        top=n,
        source=col_overlay_source,
        fill_color="white",
        fill_alpha="alpha",
        line_color=None,
        level="overlay",
    )
    fig.hbar(
        y="y",
        height=1.0,
        left=0,
        right=n,
        source=row_overlay_source,
        fill_color="white",
        fill_alpha="alpha",
        line_color=None,
        level="overlay",
    )

    modules_by_layer: dict[str, list[tuple[str, int]]] = {}
    for module_global_idx, module_name in enumerate(unique_modules):
        parts = module_name.split(".", 2)
        layer_key = f"{parts[0]}.{parts[1]}" if len(parts) >= 2 else module_name
        modules_by_layer.setdefault(layer_key, []).append((module_name, module_global_idx))

    sorted_layer_keys = sorted(modules_by_layer.keys())
    checkboxes_by_layer: dict[str, CheckboxGroup] = {}
    module_names_per_checkbox: list[list[str]] = []
    for layer_key in sorted_layer_keys:
        items = modules_by_layer[layer_key]
        labels = [name.split(".", 2)[2] if name.count(".") >= 2 else name for name, _ in items]
        cb = CheckboxGroup(
            labels=labels,
            active=list(range(len(labels))),
            inline=False,
            width=160,
        )
        checkboxes_by_layer[layer_key] = cb
        module_names_per_checkbox.append([name for name, _ in items])

    checkbox_list = [checkboxes_by_layer[k] for k in sorted_layer_keys]

    all_on_button = Button(label="All on", width=80, button_type="default")
    all_off_button = Button(label="All off", width=80, button_type="default")

    filter_callback = CustomJS(
        args=dict(
            checkboxes=checkbox_list,
            module_names_per_checkbox=module_names_per_checkbox,
            module_ids=module_ids.tolist(),
            unique_modules=list(unique_modules),
            initial_module_colors=initial_module_colors,
            color_matrix=color_matrix_js,
            slider=slider,
            col_src=col_overlay_source,
            row_src=row_overlay_source,
            module_strip_src=module_strip_source,
            cluster_strip_src=cluster_strip_source,
            whiteout_alpha=0.85,
        ),
        code="""
        const selected = new Set();
        for (let g = 0; g < checkboxes.length; g++) {
            for (const idx of checkboxes[g].active) {
                selected.add(module_names_per_checkbox[g][idx]);
            }
        }
        const sel_mask = unique_modules.map(m => selected.has(m));
        const n = module_ids.length;
        const alpha = new Array(n);
        const new_module_colors = new Array(n);
        const new_cluster_colors = new Array(n);
        const cluster_row = color_matrix[slider.value];
        const GRAY = "#d4d4d4";
        for (let i = 0; i < n; i++) {
            const is_selected = sel_mask[module_ids[i]];
            alpha[i] = is_selected ? 0.0 : whiteout_alpha;
            new_module_colors[i] = is_selected ? initial_module_colors[i] : GRAY;
            new_cluster_colors[i] = is_selected ? cluster_row[i] : GRAY;
        }
        col_src.data = {x: col_src.data.x, alpha: alpha.slice()};
        row_src.data = {y: row_src.data.y, alpha: alpha.slice()};
        const ms = module_strip_src.data;
        module_strip_src.data = {x: ms.x, y: ms.y, module_id: ms.module_id, fill_color: new_module_colors};
        const cs = cluster_strip_src.data;
        cluster_strip_src.data = {x: cs.x, y: cs.y, fill_color: new_cluster_colors};
        """,
    )
    for cb in checkbox_list:
        cb.js_on_change("active", filter_callback)

    all_on_button.js_on_click(
        CustomJS(
            args=dict(checkboxes=checkbox_list, filter_cb=filter_callback),
            code="""
            for (const cb of checkboxes) {
                cb.active = cb.labels.map((_, i) => i);
            }
            filter_cb.execute();
            """,
        )
    )
    all_off_button.js_on_click(
        CustomJS(
            args=dict(checkboxes=checkbox_list, filter_cb=filter_callback),
            code="""
            for (const cb of checkboxes) {
                cb.active = [];
            }
            filter_cb.execute();
            """,
        )
    )

    slider_callback_with_filter = CustomJS(
        args=dict(
            cluster_strip_src=cluster_strip_source,
            color_matrix=color_matrix_js,
            n_clusters_grid=n_clusters_grid_js,
            slider=slider,
            checkboxes=checkbox_list,
            module_names_per_checkbox=module_names_per_checkbox,
            module_ids=module_ids.tolist(),
            unique_modules=list(unique_modules),
        ),
        code="""
        const idx = slider.value;
        const cluster_row = color_matrix[idx];
        const selected = new Set();
        for (let g = 0; g < checkboxes.length; g++) {
            for (const a of checkboxes[g].active) {
                selected.add(module_names_per_checkbox[g][a]);
            }
        }
        const sel_mask = unique_modules.map(m => selected.has(m));
        const n = module_ids.length;
        const colors = new Array(n);
        const GRAY = "#d4d4d4";
        for (let i = 0; i < n; i++) {
            colors[i] = sel_mask[module_ids[i]] ? cluster_row[i] : GRAY;
        }
        const cs = cluster_strip_src.data;
        cluster_strip_src.data = {x: cs.x, y: cs.y, fill_color: colors};
        slider.title = "cut index (n_clusters = " + n_clusters_grid[idx] + ")";
        """,
    )
    slider.js_on_change("value", slider_callback_with_filter)

    module_filter_panel = column(
        Div(
            text=(
                "<b>Module filter</b> — uncheck modules to white them out on the heatmap "
                "and gray them in the strips."
            ),
            styles={"font-family": "sans-serif", "font-size": "12px"},
        ),
        row(all_on_button, all_off_button),
        row(
            *[
                column(
                    Div(
                        text=f"<b>{layer_key}</b>",
                        styles={"font-family": "sans-serif", "font-size": "12px"},
                    ),
                    checkboxes_by_layer[layer_key],
                )
                for layer_key in sorted_layer_keys
            ],
        ),
    )

    instructions = Div(
        text=(
            "<div style='font-family:sans-serif; font-size:12px; color:#444'>"
            "<b>Pan:</b> click & drag &nbsp;·&nbsp; "
            "<b>Zoom:</b> scroll wheel &nbsp;·&nbsp; "
            "<b>Reset:</b> click the reset icon &nbsp;·&nbsp; "
            "<b>Slider:</b> changes the dendrogram cut. "
            "<b>Module filter:</b> deselect modules to white them out on the heatmap "
            "and gray them in the strips."
            "</div>"
        ),
        width=HEAT_PIXELS + 90,
    )

    layout = column(
        instructions,
        slider,
        module_filter_panel,
        info_div,
        Div(text="<b>module strip</b>", styles={"font-family": "sans-serif", "font-size": "12px"}),
        module_strip,
        Div(
            text="<b>cluster strip (current cut)</b>",
            styles={"font-family": "sans-serif", "font-size": "12px"},
        ),
        cluster_strip,
        fig,
        sizing_mode="fixed",
    )
    return layout


def main(
    npz_path: str,
    port: int = 5179,
    address: str = "0.0.0.0",
    n_cluster_steps: int = 60,
    show: bool = False,
    save_html: str | None = None,
    heatmap_max_pixels: int = 12000,
) -> None:
    npz = Path(npz_path)
    assert npz.is_file(), f"{npz_path} does not exist"

    prepared = _Prepared(
        npz, n_cluster_steps=n_cluster_steps, heatmap_max_pixels=heatmap_max_pixels
    )

    if save_html is not None:
        from bokeh.embed import file_html
        from bokeh.resources import INLINE

        layout = _build_bokeh_layout(prepared)
        out = Path(save_html)
        html = file_html(
            layout,
            resources=INLINE,
            title="SPD coactivation viewer",
            template=LOADING_TEMPLATE,
        )
        out.write_text(html)
        logger.info(f"wrote standalone HTML to {out} ({out.stat().st_size / 1e6:.1f} MB)")
        return

    from bokeh.server.server import Server

    def app_handler(doc):
        layout = _build_bokeh_layout(prepared)
        doc.add_root(layout)
        doc.title = "SPD coactivation viewer"
        doc.template = LOADING_TEMPLATE

    logger.info(f"starting Bokeh server on {address}:{port}")
    server = Server(
        {"/": app_handler},
        port=port,
        address=address,
        allow_websocket_origin=["*"],
        websocket_max_message_size=200_000_000,
    )
    server.start()
    server.io_loop.start()


if __name__ == "__main__":
    fire.Fire(main)
