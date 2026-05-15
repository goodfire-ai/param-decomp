"""Tiny local server for viewing the U and V vectors of individual components.

Usage:
    python scripts/view_uv_matrices.py [path] [--port 8765]

Defaults to the Jose run (`wandb:goodfire/spd/s-55ea3f9b`) if no path is given.
Pick a layer + component; two stacked line plots show the sorted-descending
values of U[c, :] and V[:, c]. Use ← / → to step through components.
"""

import argparse
import threading
import webbrowser

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from param_decomp.models.component_model import ComponentModel
from param_decomp.models.components import Components

INDEX_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>U/V viewer</title>
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
  #plots { flex: 1 1 auto; display: grid; grid-template-rows: 1fr 1fr;
           gap: 8px; padding: 8px; min-height: 0; }
  .panel { position: relative; border: 1px solid #333; background: #000; min-height: 0; }
  .panel canvas { width: 100%; height: 100%; display: block; }
  .panel .title { position: absolute; top: 6px; left: 8px; color: #aaa;
                  font-size: 12px; font-family: monospace; pointer-events: none; }
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
  <span id="shape" class="meta"></span>
</header>
<div id="plots">
  <div class="panel"><div class="title" id="u-title">U</div><canvas id="u-canvas"></canvas></div>
  <div class="panel"><div class="title" id="v-title">V</div><canvas id="v-canvas"></canvas></div>
</div>

<script>
let layers = {};
let current = { layer: null, c: 0 };
let lastData = null;

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
    `U: (C=${info.C}, d_out=${info.d_out})    V: (d_in=${info.d_in}, C=${info.C})`;
  if (current.c >= info.C) current.c = 0;
  document.getElementById('comp').value = current.c;
  loadComp();
}

async function loadComp() {
  const url = `/api/component/${encodeURIComponent(current.layer)}/${current.c}`;
  const r = await fetch(url);
  lastData = await r.json();
  document.getElementById('u-title').textContent =
    `U[c, :]  sorted desc  (d_out = ${lastData.U.length})`;
  document.getElementById('v-title').textContent =
    `V[:, c]  sorted desc  (d_in = ${lastData.V.length})`;
  draw();
}

function draw() {
  if (!lastData) return;
  drawPlot('u-canvas', lastData.U);
  drawPlot('v-canvas', lastData.V);
}

function drawPlot(id, values) {
  const cvs = document.getElementById(id);
  const dpr = window.devicePixelRatio || 1;
  const cssW = cvs.clientWidth, cssH = cvs.clientHeight;
  cvs.width = Math.max(1, Math.floor(cssW * dpr));
  cvs.height = Math.max(1, Math.floor(cssH * dpr));
  const ctx = cvs.getContext('2d');
  ctx.scale(dpr, dpr);

  const pad = { l: 56, r: 12, t: 24, b: 24 };
  const W = cssW - pad.l - pad.r;
  const H = cssH - pad.t - pad.b;

  ctx.clearRect(0, 0, cssW, cssH);

  const n = values.length;
  if (n === 0 || W <= 0 || H <= 0) return;
  let vmin = values[n - 1], vmax = values[0];
  if (vmin === vmax) { vmin -= 1; vmax += 1; }
  // Include 0 in range so the zero crossing is visible.
  vmin = Math.min(vmin, 0);
  vmax = Math.max(vmax, 0);

  const xOf = i => pad.l + (i / Math.max(1, n - 1)) * W;
  const yOf = v => pad.t + (1 - (v - vmin) / (vmax - vmin)) * H;

  // Axes
  ctx.strokeStyle = '#444';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, pad.t + H);
  ctx.lineTo(pad.l + W, pad.t + H);
  ctx.stroke();

  // Zero line
  if (vmin < 0 && vmax > 0) {
    const y0 = yOf(0);
    ctx.strokeStyle = '#555';
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(pad.l, y0); ctx.lineTo(pad.l + W, y0);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Tick labels
  ctx.fillStyle = '#aaa';
  ctx.font = '11px monospace';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  ctx.fillText(vmax.toExponential(2), pad.l - 4, pad.t);
  ctx.fillText(vmin.toExponential(2), pad.l - 4, pad.t + H);
  if (vmin < 0 && vmax > 0) ctx.fillText('0', pad.l - 4, yOf(0));

  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.fillText('rank 0', pad.l, pad.t + H + 4);
  ctx.textAlign = 'right';
  ctx.fillText(`rank ${n - 1}`, pad.l + W, pad.t + H + 4);

  // Line
  ctx.strokeStyle = '#5fd';
  ctx.lineWidth = 1.25;
  ctx.beginPath();
  // Sub-sample if there are way more points than pixels.
  const step = Math.max(1, Math.floor(n / (W * dpr * 2)));
  for (let i = 0; i < n; i += step) {
    const x = xOf(i), y = yOf(values[i]);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  // Always include the last point.
  ctx.lineTo(xOf(n - 1), yOf(values[n - 1]));
  ctx.stroke();
}

document.getElementById('layer').addEventListener('change', e => {
  current.layer = e.target.value;
  updateLayer();
});
document.getElementById('comp').addEventListener('change', e => {
  current.c = parseInt(e.target.value, 10) || 0;
  loadComp();
});
document.getElementById('prev').addEventListener('click', () => {
  current.c = Math.max(0, current.c - 1);
  document.getElementById('comp').value = current.c;
  loadComp();
});
document.getElementById('next').addEventListener('click', () => {
  const max = layers[current.layer].C - 1;
  current.c = Math.min(max, current.c + 1);
  document.getElementById('comp').value = current.c;
  loadComp();
});
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  if (e.key === 'ArrowLeft') document.getElementById('prev').click();
  if (e.key === 'ArrowRight') document.getElementById('next').click();
});
window.addEventListener('resize', draw);

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


def make_app(model: ComponentModel) -> FastAPI:
    components = build_inventory(model)
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    meta: dict[str, dict[str, int]] = {}
    for name, comp in components.items():
        U = comp.U.detach().float().cpu().numpy()  # (C, d_out)
        V = comp.V.detach().float().cpu().numpy()  # (d_in, C)
        cache[name] = (U, V)
        meta[name] = {"C": int(U.shape[0]), "d_in": int(V.shape[0]), "d_out": int(U.shape[1])}

    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return INDEX_HTML

    @app.get("/api/layers")
    def layers() -> JSONResponse:
        return JSONResponse(meta)

    @app.get("/api/component/{layer}/{c}")
    def component(layer: str, c: int) -> JSONResponse:
        if layer not in cache:
            raise HTTPException(404, f"unknown layer {layer}")
        U, V = cache[layer]
        if not (0 <= c < U.shape[0]):
            raise HTTPException(404, f"component {c} out of range [0, {U.shape[0]})")
        u_sorted = np.sort(U[c])[::-1].tolist()
        v_sorted = np.sort(V[:, c])[::-1].tolist()
        return JSONResponse({"U": u_sorted, "V": v_sorted})

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
