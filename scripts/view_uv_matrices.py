"""Tiny local server for viewing materialized U@V matrices of individual components.

Usage:
    python scripts/view_uv_matrices.py [path] [--port 8765]

Defaults to the Jose run (`wandb:goodfire/spd/s-55ea3f9b`) if no path is given.
Then open http://localhost:8765 in a browser. Pick a layer + component index;
the per-component outer product V[:, c] @ U[c, :] (in (d_out, d_in) layout matching
nn.Linear.weight) is rendered as a PNG heatmap. Drag/zoom via the magnifier lens.
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
  body { font-family: system-ui, sans-serif; margin: 16px; background: #111; color: #eee; }
  header { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin-bottom: 12px; }
  select, input, button { font-size: 14px; padding: 4px 8px; background: #222; color: #eee;
                          border: 1px solid #444; border-radius: 4px; }
  button:hover { background: #333; cursor: pointer; }
  .meta { color: #888; font-size: 12px; margin-left: 8px; }
  #stage { position: relative; display: inline-block; max-width: 100%; max-height: 80vh;
           overflow: auto; border: 1px solid #333; background: #000; }
  #img { display: block; image-rendering: pixelated; }
  #lens { position: absolute; pointer-events: none; border: 2px solid #ff0;
          width: 220px; height: 220px; display: none; overflow: hidden;
          box-shadow: 0 0 0 1px #000; background-repeat: no-repeat; }
  #cursor-info { color: #888; font-size: 12px; margin-top: 6px; font-family: monospace; }
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
    <input id="zoom" type="range" min="2" max="20" value="8" />
    <span id="zoomVal" class="meta">8x</span>
  </label>
  <span id="shape" class="meta"></span>
</header>
<div id="stage">
  <img id="img" />
  <div id="lens"></div>
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
  document.getElementById('shape').textContent = `${info.rows} × ${info.cols} (d_out × d_in)`;
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
  // Position lens inside stage (account for scroll)
  const stageRect = stage.getBoundingClientRect();
  const sx = e.clientX - stageRect.left + stage.scrollLeft - lensW / 2;
  const sy = e.clientY - stageRect.top + stage.scrollTop - lensH / 2;
  lens.style.left = sx + 'px';
  lens.style.top = sy + 'px';

  // Map mouse to native image coords
  const natX = (x / rect.width) * img.naturalWidth;
  const natY = (y / rect.height) * img.naturalHeight;
  const bgW = img.naturalWidth * zoom;
  const bgH = img.naturalHeight * zoom;
  lens.style.backgroundSize = `${bgW}px ${bgH}px`;
  lens.style.backgroundPosition =
    `${-(natX * zoom - lensW / 2)}px ${-(natY * zoom - lensH / 2)}px`;

  // Show pixel coordinates: row=d_out, col=d_in
  document.getElementById('cursor-info').textContent =
    `row (d_out) = ${Math.floor(natY)}   col (d_in) = ${Math.floor(natX)}`;
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


def render_png(U_row: np.ndarray, V_col: np.ndarray) -> bytes:
    """U_row: (d_out,), V_col: (d_in,). Returns PNG of outer = V_col[:, None] @ U_row[None, :]
    laid out as (d_out, d_in) to match nn.Linear.weight convention."""
    # outer[i, j] = V_col[j] * U_row[i]  -> shape (d_out, d_in)
    outer = np.outer(U_row, V_col)
    vmax = float(np.abs(outer).max())
    if vmax == 0.0:
        vmax = 1.0
    norm = (outer / vmax + 1.0) * 0.5  # map [-vmax, vmax] -> [0, 1]
    cmap = cm.get_cmap("RdBu_r")
    rgba = (cmap(norm) * 255).astype(np.uint8)
    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False, compress_level=1)
    return buf.getvalue()


def make_app(model: ComponentModel) -> FastAPI:
    components = build_inventory(model)
    # Stash U, V as cpu float32 numpy arrays for fast slicing.
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    meta: dict[str, dict[str, int]] = {}
    for name, comp in components.items():
        U = comp.U.detach().float().cpu().numpy()  # (C, d_out)
        V = comp.V.detach().float().cpu().numpy()  # (d_in, C)
        cache[name] = (U, V)
        meta[name] = {"C": int(U.shape[0]), "rows": int(U.shape[1]), "cols": int(V.shape[0])}

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
        return render_png(U[c], V[:, c])

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
