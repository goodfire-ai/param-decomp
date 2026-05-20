"""Render PhaseProfiler JSON dumps as a stacked-track HTML timeline.

Reads JSON files written by `two_pool_wider.py` under PROFILE_OUT_DIR (one per
(mode, pool, rank)) and emits a single HTML page with two stacked tracks per
step (pool A on top, pool B below). One section per mode (sync / async).

Usage:
    python -m param_decomp.scripts.two_pool_benchmark.make_profile_html \\
        --in /mnt/polished-lake/home/oli/two_pool_profile \\
        --out /mnt/polished-lake/home/oli/two_pool_profile/profile.html
"""

# pyright: reportArgumentType=false

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def color_for(name: str) -> str:
    """Stable HSL color per phase name."""
    h = int(hashlib.md5(name.encode()).hexdigest(), 16)
    hue = h % 360
    return f"hsl({hue}, 60%, 55%)"


def render_track(
    spans: list[dict],
    *,
    pool: str,
    max_ms: float,
    px_per_ms: float,
    track_height: int,
) -> str:
    """One horizontal track of spans for a single (mode, step, pool)."""
    width_px = max(1, int(max_ms * px_per_ms)) + 20
    pool_label = "pool A" if pool == "a" else "pool B"
    color = "#1f77b4" if pool == "a" else "#ff7f0e"
    track_inner_h = track_height - 4
    out: list[str] = []
    out.append(
        f'<div class="track-row">'
        f'<div class="track-label" style="border-color:{color}">{pool_label}</div>'
        f'<div class="track" style="width:{width_px}px;height:{track_height}px;">'
    )
    for s in spans:
        left = int(s["start_ms"] * px_per_ms)
        w = max(1, int((s["end_ms"] - s["start_ms"]) * px_per_ms))
        c = color_for(s["name"])
        duration = s["end_ms"] - s["start_ms"]
        title = f'{s["name"]}: {s["start_ms"]:.2f}–{s["end_ms"]:.2f} ms ({duration:.2f} ms)'
        # Phase name printed inline if box is wide enough.
        label = s["name"].split("/", 1)[-1] if w > 60 else ""
        out.append(
            f'<div class="span" style="left:{left}px;width:{w}px;height:{track_inner_h}px;'
            f'background:{c};" title="{title}"><span class="span-label">{label}</span></div>'
        )
    out.append('</div></div>')
    return "".join(out)


def render_axis(max_ms: float, px_per_ms: float) -> str:
    """X-axis tick marks every 100ms."""
    width_px = max(1, int(max_ms * px_per_ms)) + 20
    tick_step = 100
    ticks: list[str] = []
    t = 0
    while t <= max_ms:
        left = int(t * px_per_ms)
        ticks.append(f'<div class="tick" style="left:{left}px;">{t}ms</div>')
        t += tick_step
    return (
        f'<div class="axis-row"><div class="track-label"></div>'
        f'<div class="axis" style="width:{width_px}px;">{"".join(ticks)}</div></div>'
    )


def render_step(
    *, step: int, a_spans: list[dict], b_spans: list[dict],
    max_ms: float, px_per_ms: float,
) -> str:
    return (
        f'<div class="step"><h3>step {step}</h3>'
        f'{render_axis(max_ms, px_per_ms)}'
        f'{render_track(a_spans, pool="a", max_ms=max_ms, px_per_ms=px_per_ms, track_height=28)}'
        f'{render_track(b_spans, pool="b", max_ms=max_ms, px_per_ms=px_per_ms, track_height=28)}'
        f'</div>'
    )


def render_mode_section(
    *, mode: str, pool_a: dict, pool_b: dict, px_per_ms: float,
) -> str:
    by_step: dict[int, dict[str, list[dict]]] = defaultdict(lambda: {"a": [], "b": []})
    for s in pool_a["spans"]:
        by_step[s["step"]]["a"].append(s)
    for s in pool_b["spans"]:
        by_step[s["step"]]["b"].append(s)
    max_ms = max(
        (s["end_ms"] for spans in by_step.values() for s in spans["a"] + spans["b"]),
        default=0.0,
    )
    desc = {
        "sync": (
            "cuda.synchronize() between phases. Each phase's duration is its honest "
            "wall-clock cost, but all overlap is gone — async sends look as expensive "
            "as their underlying work. Total step time is inflated."
        ),
        "async": (
            "perf_counter() only — no synchronization. Spans show when Python was IN "
            "each phase block. Async sends/recvs appear tiny because the work continues "
            "off-thread; the wait shows up later in a different phase."
        ),
    }[mode]
    steps_html = "".join(
        render_step(
            step=step, a_spans=by_step[step]["a"], b_spans=by_step[step]["b"],
            max_ms=max_ms, px_per_ms=px_per_ms,
        )
        for step in sorted(by_step.keys())
    )
    return (
        f'<section class="mode-section"><h2>profiler mode: {mode}</h2>'
        f'<p class="mode-desc">{desc}</p>{steps_html}</section>'
    )


CSS = """
body { font-family: -apple-system, sans-serif; margin: 24px; background: #fafafa; color: #222; }
h1 { margin: 0 0 8px 0; }
.mode-section { margin: 32px 0; padding: 16px; background: white; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
.mode-section h2 { margin-top: 0; }
.mode-desc { color: #555; font-size: 14px; max-width: 800px; line-height: 1.5; }
.step { margin: 16px 0 24px; }
.step h3 { margin: 0 0 4px 0; font-weight: 500; color: #444; }
.axis-row { display: flex; align-items: center; margin: 4px 0; }
.track-row { display: flex; align-items: center; margin: 2px 0; }
.track-label {
  width: 70px; padding: 4px 8px; font-size: 12px; color: #444;
  border-left: 4px solid #ccc; box-sizing: border-box;
}
.track { position: relative; background: #f0f0f0; border: 1px solid #ddd; box-sizing: border-box; }
.axis { position: relative; height: 16px; }
.tick {
  position: absolute; top: 0; font-size: 10px; color: #888;
  border-left: 1px solid #ccc; padding-left: 2px; height: 100%; box-sizing: border-box;
}
.span {
  position: absolute; top: 2px; border-radius: 2px; cursor: pointer;
  overflow: hidden; box-sizing: border-box; opacity: 0.92;
}
.span:hover { opacity: 1; box-shadow: 0 0 0 1px #222; z-index: 10; }
.span-label {
  display: block; padding: 4px 6px; color: white; font-size: 11px;
  font-family: ui-monospace, monospace; white-space: nowrap; text-shadow: 0 1px 2px rgba(0,0,0,0.4);
}
.legend { margin-top: 8px; font-size: 12px; color: #666; }
"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--px-per-ms", type=float, default=1.4)
    args = p.parse_args()

    by_mode: dict[str, dict[str, dict]] = defaultdict(dict)
    for path in sorted(args.in_dir.glob("wider_*.json")):
        data = json.loads(path.read_text())
        by_mode[data["mode"]][data["pool"]] = data

    sections: list[str] = []
    for mode in ("sync", "async"):
        if mode not in by_mode:
            continue
        if "a" not in by_mode[mode] or "b" not in by_mode[mode]:
            print(f"[warn] mode={mode} missing pool data — got {list(by_mode[mode].keys())}")
            continue
        sections.append(render_mode_section(
            mode=mode, pool_a=by_mode[mode]["a"], pool_b=by_mode[mode]["b"],
            px_per_ms=args.px_per_ms,
        ))

    html = (
        '<!doctype html><html><head><meta charset="utf-8"><title>2-pool profile</title>'
        f'<style>{CSS}</style></head><body>'
        '<h1>2-pool wider — phase timeline</h1>'
        '<p style="color:#666;max-width:800px;line-height:1.5;">'
        'Each step shows pool A (top track) and pool B (bottom track) over the same '
        'time axis. Both pools are synchronized at step start via dist.barrier so the '
        'tracks share a common origin. Hover for exact ms; box color is per phase name.'
        '</p>'
        f'{"".join(sections)}'
        '</body></html>'
    )
    args.out.write_text(html)
    print(f"wrote {args.out} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
