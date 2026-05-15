"""Tiny local server for viewing materialized U@V matrices of individual components.

Usage:
    python scripts/view_uv_matrices.py [path] [--port 8765]

Defaults to the Jose run (`wandb:goodfire/spd/s-55ea3f9b`) if no path is given.
Then open http://localhost:8765 in a browser. Pick a layer + component index;
the per-component outer product V[:, c] @ U[c, :] (semantically (d_out, d_in)
matching nn.Linear.weight) is rendered as a PNG heatmap. The image is
auto-oriented so the longer axis is horizontal. Drag/zoom via the magnifier lens.
"""

import argparse
import io
import threading
import webbrowser
from functools import lru_cache

import matplotlib.cm as cm
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from PIL import Image

from param_decomp.models.component_model import ComponentModel
from param_decomp.models.components import Components

INDEX_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>U@V viewer</title>
<style>
  html, body { height: 100%; margin: 0; }
  body { font-family: system-ui, sans-serif; background: #111; color: #eee;
         display: flex; flex-direction: column; }
  header { display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
           padding: 8px 12px; border-bottom: 1px solid #333; }
  select, input, button { font-size: 14px; padding: 4px 8px; background: #222; color: #eee;
                          border: 1px solid #444; border-radius: 4px; }
  button:hover { background: #333; cursor: pointer; }
  .meta { color: #888; font-size: 12px; margin-left: 8px; }

  /* Grid: [corner | x-label] / [y-label | stage] */
  #viewer { flex: 1 1 auto; display: grid;
            grid-template-columns: 28px 1fr;
            grid-template-rows: 22px 1fr;
            min-height: 0; }
  #corner { grid-column: 1; grid-row: 1; }
  #x-axis { grid-column: 2; grid-row: 1;
            display: flex; align-items: center; justify-content: center;
            color: #aaa; font-size: 12px; font-family: monospace; }
  #y-axis { grid-column: 1; grid-row: 2;
            display: flex; align-items: center; justify-content: center;
            color: #aaa; font-size: 12px; font-family: monospace;
            writing-mode: vertical-rl; transform: rotate(180deg); }
  #stage { grid-column: 2; grid-row: 2;
           position: relative; overflow: auto;
           border: 1px solid #333; background: #000; min-width: 0; min-height: 0; }
  #img { display: block; image-rendering: pixelated; image-rendering: crisp-edges; }
  #lens { position: absolute; pointer-events: none; border: 2px solid #ff0;
          width: 280px; height: 280px; display: none; overflow: hidden;
          box-shadow: 0 0 0 1px #000; background-repeat: no-repeat;
          image-rendering: pixelated; image-rendering: crisp-edges; }
  #cursor-info { padding: 4px 12px; color: #888; font-size: 12px; font-family: monospace;
                 border-top: 1px solid #333; }
</style>
</head>
<body>
<header>
  <label>layer
    <select id="layer"></select>
  </label>
  <label>component
    <button id="prev">&larr;</button>
    <input id="comp" type="number" min="0" value="0" style="width: 80px" />
    <button id="next">&rarr;</button>
    <span id="cmax" class="meta"></span>
  </label>
  <label>zoom
    <input id="zoom" type="range" min="2" max="32" value="8" />
    <span id="zoomVal" class="meta">8x</span>
  </label>
  <span id="shape" class="meta"></span>
</header>
<div id="viewer">
  <div id="corner"></div>
  <div id="x-axis"></div>
  <div id="y-axis"></div>
  <div id="stage">
    <img id="img" />
    <div id="lens"></div>
  </div>
</div>
<div id="cursor-info"></div>

<script>
let layers = {};
let current = { layer: null, c: 0 };

async function loadLayers() {
  const r = await fetch('/api/layers');
  layers = await r.json();
  const sel = document.getElementById('layer');
  sel.innerHTML = '';
  for (const name of Object.keys(layers)) {
    const o = document.createElement('option');
    o.value = name; o.textContent = name;
    sel.appendChild(o);
  }
  sel.value = Object.keys(layers)[0];
  current.layer = sel.value;
  updateLayer();
}

function updateLayer() {
  const info = layers[current.layer];
  document.getElementById('comp').max = info.C - 1;
  document.getElementById('cmax').textContent = `/ ${info.C - 1}`;
  document.getElementById('shape').textContent =
    `image ${info.width}×${info.height}  (rows d_out=${info.d_out}, cols d_in=${info.d_in})`;
  document.getElementById('x-axis').textContent = `${info.x_label} (${info.width})`;
  document.getElementById('y-axis').textContent = `${info.y_label} (${info.height})`;
  if (current.c >= info.C) current.c = 0;
  document.getElementById('comp').value = current.c;
  loadMatrix();
}

function loadMatrix() {
  const url = `/api/matrix/${encodeURIComponent(current.layer)}/${current.c}.png?t=${Date.now()}`;
  const img = document.getElementById('img');
  img.src = url;
  const lens = document.getElementById('lens');
  lens.style.backgroundImage = `url('${url}')`;
}

document.getElementById('layer').addEventListener('change', e => {
  current.layer = e.target.value;
  updateLayer();
});
document.getElementById('comp').addEventListener('change', e => {
  current.c = parseInt(e.target.value, 10) || 0;
  loadMatrix();
});
document.getElementById('prev').addEventListener('click', () => {
  current.c = Math.max(0, current.c - 1);
  document.getElementById('comp').value = current.c;
  loadMatrix();
});
document.getElementById('next').addEventListener('click', () => {
  const max = layers[current.layer].C - 1;
  current.c = Math.min(max, current.c + 1);
  document.getElementById('comp').value = current.c;
  loadMatrix();
});
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  if (e.key === 'ArrowLeft') document.getElementById('prev').click();
  if (e.key === 'ArrowRight') document.getElementById('next').click();
});

const zoomEl = document.getElementById('zoom');
const zoomVal = document.getElementById('zoomVal');
zoomEl.addEventListener('input', () => {
  zoomVal.textContent = `${zoomEl.value}x`;
});

const stage = document.getElementById('stage');
const img = document.getElementById('img');
const lens = document.getElementById('lens');

img.addEventListener('mouseenter', () => { lens.style.display = 'block'; });
img.addEventListener('mouseleave', () => { lens.style.display = 'none'; });
img.addEventListener('mousemove', e => {
  const rect = img.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  const zoom = parseFloat(zoomEl.value);

  const lensW = lens.offsetWidth, lensH = lens.offsetHeight;
  const stageRect = stage.getBoundingClientRect();
  const sx = e.clientX - stageRect.left + stage.scrollLeft - lensW / 2;
  const sy = e.clientY - stageRect.top + stage.scrollTop - lensH / 2;
  lens.style.left = sx + 'px';
  lens.style.top = sy + 'px';

  // Map mouse to native image pixel coords
  const natX = (x / rect.width) * img.naturalWidth;
  const natY = (y / rect.height) * img.naturalHeight;
  // Background is the image scaled by `zoom` relative to its native pixels,
  // so the lens always samples native pixels (no resampling from the displayed size).
  const bgW = img.naturalWidth * zoom;
  const bgH = img.naturalHeight * zoom;
  lens.style.backgroundSize = `${bgW}px ${bgH}px`;
  lens.style.backgroundPosition =
    `${-(natX * zoom - lensW / 2)}px ${-(natY * zoom - lensH / 2)}px`;

  const info = layers[current.layer];
  document.getElementById('cursor-info').textContent =
    `${info.x_label} = ${Math.floor(natX)}    ${info.y_label} = ${Math.floor(natY)}`;
});

loadLayers();
</script>
</body>
</html>
"""


def build_inventory(model: ComponentModel) -> dict[str, Components]:
    inv: dict[str, Components] = {}
    for name, comp in model.components.items():
        assert hasattr(comp, "U") and hasattr(comp, "V"), f"{name} has no U/V"
        inv[name] = comp
    return inv


def render_png(U_row: np.ndarray, V_col: np.ndarray, transpose: bool) -> bytes:
    """U_row: (d_out,), V_col: (d_in,). Returns PNG of outer = U_row[:, None] * V_col[None, :].

    Semantic shape is (d_out, d_in) matching nn.Linear.weight. If `transpose` is True,
    we rotate so columns are the longer dimension before encoding.
    """
    outer = np.outer(U_row, V_col)  # (d_out, d_in)
    if transpose:
        outer = outer.T
    vmax = float(np.abs(outer).max())
    if vmax == 0.0:
        vmax = 1.0
    norm = (outer / vmax + 1.0) * 0.5
    cmap = cm.get_cmap("RdBu_r")
    rgba = (cmap(norm) * 255).astype(np.uint8)
    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False, compress_level=1)
    return buf.getvalue()


def make_app(model: ComponentModel) -> FastAPI:
    components = build_inventory(model)
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    transpose_by_layer: dict[str, bool] = {}
    meta: dict[str, dict[str, object]] = {}
    for name, comp in components.items():
        U = comp.U.detach().float().cpu().numpy()  # (C, d_out)
        V = comp.V.detach().float().cpu().numpy()  # (d_in, C)
        C, d_out = int(U.shape[0]), int(U.shape[1])
        d_in = int(V.shape[0])
        # Orient so the wider axis becomes the image width.
        transpose = d_out > d_in
        width, height = (d_out, d_in) if transpose else (d_in, d_out)
        x_label, y_label = ("d_out", "d_in") if transpose else ("d_in", "d_out")
        cache[name] = (U, V)
        transpose_by_layer[name] = transpose
        meta[name] = {
            "C": C,
            "d_in": d_in,
            "d_out": d_out,
            "width": width,
            "height": height,
            "x_label": x_label,
            "y_label": y_label,
        }

    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return INDEX_HTML

    @app.get("/api/layers")
    def layers() -> JSONResponse:
        return JSONResponse(meta)

    @lru_cache(maxsize=64)
    def get_png(layer: str, c: int) -> bytes:
        if layer not in cache:
            raise HTTPException(404, f"unknown layer {layer}")
        U, V = cache[layer]
        if not (0 <= c < U.shape[0]):
            raise HTTPException(404, f"component {c} out of range [0, {U.shape[0]})")
        return render_png(U[c], V[:, c], transpose_by_layer[layer])

    @app.get("/api/matrix/{layer}/{c}.png")
    def matrix(layer: str, c: int) -> Response:
        return Response(content=get_png(layer, c), media_type="image/png")

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        default="wandb:goodfire/spd/s-55ea3f9b",
        help="wandb:entity/project/run_id or local checkpoint path (default: Jose run)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    print(f"Loading ComponentModel from {args.path} ...")
    with torch.no_grad():
        model = ComponentModel.from_pretrained(args.path)
    print(f"Loaded. Components: {list(model.components.keys())}")
    url = f"http://{args.host}:{args.port}"
    print(f"Serving on {url}")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(make_app(model), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
