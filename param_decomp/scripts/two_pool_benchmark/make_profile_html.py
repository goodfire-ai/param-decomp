"""Render PhaseProfiler JSON dumps as a stacked-track HTML timeline.

Reads JSON files written by `two_pool_wider.py` under PROFILE_OUT_DIR (one per
(mode, pool, rank)) and emits a single HTML page with two stacked tracks per
step (pool A on top, pool B below). One section per mode (sync / async).

Designed for exactness over prettiness:
  - Every phase label is rendered, even on narrow boxes (label can extend past
    the box, becomes draggable to front on hover via z-index).
  - Every label includes its duration in ms.
  - A full data table accompanies each step with start/end/duration for every
    span.

Usage:
    python -m param_decomp.scripts.two_pool_benchmark.make_profile_html \\
        --in /mnt/polished-lake/home/oli/two_pool_profile \\
        --out docs/two_pool_timeline_profile.html
"""

# pyright: reportArgumentType=false, reportMissingTypeArgument=false
# pyright: reportUnknownParameterType=false, reportMissingParameterType=false
# pyright: reportUnknownArgumentType=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false, reportUnknownLambdaType=false

import argparse
import hashlib
import html as _html
import json
from collections import defaultdict
from pathlib import Path


def color_for(name: str) -> str:
    """Stable HSL color per phase name."""
    h = int(hashlib.md5(name.encode()).hexdigest(), 16)
    hue = h % 360
    return f"hsl({hue}, 60%, 50%)"


def short_name(full: str) -> str:
    """Drop the 'a/' or 'b/' prefix when displaying in-box."""
    return full.split("/", 1)[-1] if "/" in full else full


# Phase taxonomy: pattern-matched on the trailing part of the phase name. Used
# to (a) color a category stripe at the bottom of each box, (b) compute the
# active/waiting/comm-time summary per pool. Update when phase names change.
CATEGORY_RULES = (
    ("wait", ("wait_", "_wait")),
    ("comm", ("send_", "recv_", "allreduce", "async_send", "post_async")),
    ("compute", ()),  # default catch-all
)
CATEGORY_COLORS = {
    "compute": "#2ca02c",  # green
    "comm": "#d62728",  # red
    "wait": "#7f7f7f",  # gray
}


def categorize(name: str) -> str:
    short = short_name(name)
    for cat, patterns in CATEGORY_RULES:
        if any(pat in short for pat in patterns):
            return cat
    return "compute"


def render_track(
    spans: list[dict],
    *,
    pool: str,
    max_ms: float,
    px_per_ms: float,
    track_height: int,
) -> str:
    """One horizontal track of spans for a single (mode, step, pool).

    Labels are always rendered. Narrow spans get the label outside the box
    (overflow visible) — they'll overlap neighbours visually but hover
    brings them forward.
    """
    width_px = max(1, int(max_ms * px_per_ms)) + 60
    pool_label = "pool A" if pool == "a" else "pool B"
    pool_color = "#1f77b4" if pool == "a" else "#ff7f0e"
    track_inner_h = track_height - 4
    out: list[str] = []
    out.append(
        f'<div class="track-row">'
        f'<div class="track-label" style="border-color:{pool_color}">{pool_label}</div>'
        f'<div class="track" style="width:{width_px}px;height:{track_height}px;">'
    )
    for s in sorted(spans, key=lambda x: x["start_ms"]):
        left = s["start_ms"] * px_per_ms
        w = max(0.5, (s["end_ms"] - s["start_ms"]) * px_per_ms)
        c = color_for(s["name"])
        duration = s["end_ms"] - s["start_ms"]
        cat = categorize(s["name"])
        cat_c = CATEGORY_COLORS[cat]
        title = (
            f"{s['name']} [{cat}]\n"
            f"start: {s['start_ms']:.3f} ms\n"
            f"end:   {s['end_ms']:.3f} ms\n"
            f"duration: {duration:.3f} ms"
        )
        label = f"{short_name(s['name'])} ({duration:.2f}ms)"
        out.append(
            f'<div class="span" style="left:{left:.2f}px;width:{w:.2f}px;'
            f"height:{track_inner_h}px;background:{c};"
            f'border-bottom:3px solid {cat_c};" '
            f'title="{_html.escape(title)}">'
            f'<span class="span-label">{_html.escape(label)}</span>'
            f"</div>"
        )
    out.append("</div></div>")
    return "".join(out)


def render_axis(max_ms: float, px_per_ms: float) -> str:
    """X-axis tick marks every 50ms."""
    width_px = max(1, int(max_ms * px_per_ms)) + 60
    tick_step = 50
    ticks: list[str] = []
    t = 0
    while t <= max_ms + tick_step:
        left = t * px_per_ms
        ticks.append(f'<div class="tick" style="left:{left:.2f}px;">{t}ms</div>')
        t += tick_step
    return (
        f'<div class="axis-row"><div class="track-label"></div>'
        f'<div class="axis" style="width:{width_px}px;">{"".join(ticks)}</div></div>'
    )


def render_span_table(spans: list[dict], *, pool: str) -> str:
    """A precise tabular dump of every span: name | start | end | duration."""
    rows: list[str] = []
    for s in sorted(spans, key=lambda x: x["start_ms"]):
        duration = s["end_ms"] - s["start_ms"]
        c = color_for(s["name"])
        rows.append(
            f"<tr>"
            f'<td><span class="swatch" style="background:{c}"></span>{_html.escape(s["name"])}</td>'
            f'<td class="num">{s["start_ms"]:.3f}</td>'
            f'<td class="num">{s["end_ms"]:.3f}</td>'
            f'<td class="num">{duration:.3f}</td>'
            f"</tr>"
        )
    return (
        f'<table class="span-table"><caption>{("pool A" if pool == "a" else "pool B")} '
        f"spans</caption><thead><tr><th>phase</th><th>start (ms)</th>"
        f"<th>end (ms)</th><th>duration (ms)</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def per_pool_breakdown(spans: list[dict]) -> dict[str, float]:
    """Aggregate phase durations by category for one pool over one step."""
    totals: dict[str, float] = {"compute": 0.0, "comm": 0.0, "wait": 0.0}
    for s in spans:
        totals[categorize(s["name"])] += s["end_ms"] - s["start_ms"]
    return totals


def render_step_summary(*, a_spans: list[dict], b_spans: list[dict]) -> str:
    """Headline metrics box: total wall-clock + per-pool compute/comm/wait split."""
    a = per_pool_breakdown(a_spans)
    b = per_pool_breakdown(b_spans)
    a_total = max((s["end_ms"] for s in a_spans), default=0.0)
    b_total = max((s["end_ms"] for s in b_spans), default=0.0)
    step_total = max(a_total, b_total)

    def fmt(label: str, total: float, br: dict[str, float]) -> str:
        # Categories sum to 'total' (modulo profiler overhead) but Python's view
        # in async mode can have gaps; render as both raw ms and %.
        parts = []
        for cat in ("compute", "comm", "wait"):
            v = br[cat]
            pct = (100 * v / total) if total > 0 else 0
            parts.append(
                f'<span class="cat-pill" style="background:{CATEGORY_COLORS[cat]}">'
                f"{cat} {v:.1f}ms ({pct:.0f}%)</span>"
            )
        return (
            f'<div class="pool-summary"><span class="pool-name">{label}</span>'
            f'<span class="total">total {total:.1f}ms</span>{"".join(parts)}</div>'
        )

    return (
        f'<div class="step-summary">'
        f'<div class="step-total">step wall-clock: <strong>{step_total:.1f} ms</strong></div>'
        f"{fmt('pool A', a_total, a)}{fmt('pool B', b_total, b)}"
        f"</div>"
    )


def render_step(
    *,
    step: int,
    a_spans: list[dict],
    b_spans: list[dict],
    max_ms: float,
    px_per_ms: float,
) -> str:
    return (
        f'<div class="step"><h3>step {step}</h3>'
        f"{render_step_summary(a_spans=a_spans, b_spans=b_spans)}"
        f'<div class="timeline-wrap">'
        f"{render_axis(max_ms, px_per_ms)}"
        f"{render_track(a_spans, pool='a', max_ms=max_ms, px_per_ms=px_per_ms, track_height=32)}"
        f"{render_track(b_spans, pool='b', max_ms=max_ms, px_per_ms=px_per_ms, track_height=32)}"
        f"</div>"
        f'<details class="span-tables"><summary>raw spans (click)</summary>'
        f'<div class="tables-row">{render_span_table(a_spans, pool="a")}'
        f"{render_span_table(b_spans, pool='b')}</div>"
        f"</details>"
        f"</div>"
    )


def render_summary_table(*, pool_a: dict, pool_b: dict) -> str:
    """Per-phase mean duration over all profiled steps."""
    rows_html: list[str] = []
    for pool_name, data in (("pool A", pool_a), ("pool B", pool_b)):
        agg: dict[str, list[float]] = defaultdict(list)
        for s in data["spans"]:
            agg[s["name"]].append(s["end_ms"] - s["start_ms"])
        for name, durations in sorted(agg.items()):
            c = color_for(name)
            avg = sum(durations) / len(durations)
            rows_html.append(
                f"<tr>"
                f"<td>{pool_name}</td>"
                f'<td><span class="swatch" style="background:{c}"></span>{_html.escape(name)}</td>'
                f'<td class="num">{avg:.3f}</td>'
                f'<td class="num">{min(durations):.3f}</td>'
                f'<td class="num">{max(durations):.3f}</td>'
                f'<td class="num">{len(durations)}</td>'
                f"</tr>"
            )
    return (
        f'<table class="span-table summary"><caption>per-phase aggregates '
        f"(across profiled steps)</caption><thead><tr><th>pool</th><th>phase</th>"
        f"<th>avg (ms)</th><th>min (ms)</th><th>max (ms)</th><th>n</th></tr>"
        f"</thead><tbody>{''.join(rows_html)}</tbody></table>"
    )


def render_mode_section(
    *,
    mode: str,
    pool_a: dict,
    pool_b: dict,
    px_per_ms: float,
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
            "perf_counter() only — no synchronization between phases. Spans show "
            "when Python was IN each phase block. NCCL irecv .wait() returns once "
            "the op is enqueued on the GPU stream, not once data arrives — so the "
            "extra b/3a_wait_ci_recv_gpu phase forces a cuda.synchronize() to make "
            "the GPU-side wait visible."
        ),
    }[mode]
    steps_html = "".join(
        render_step(
            step=step,
            a_spans=by_step[step]["a"],
            b_spans=by_step[step]["b"],
            max_ms=max_ms,
            px_per_ms=px_per_ms,
        )
        for step in sorted(by_step.keys())
    )
    summary = render_summary_table(pool_a=pool_a, pool_b=pool_b)
    return (
        f'<section class="mode-section"><h2>profiler mode: {mode}</h2>'
        f'<p class="mode-desc">{desc}</p>{steps_html}'
        f'<details open class="mode-summary"><summary>per-phase aggregates</summary>'
        f"{summary}</details>"
        f"</section>"
    )


CSS = """
body { font-family: -apple-system, sans-serif; margin: 24px; background: #fafafa; color: #222; }
h1 { margin: 0 0 8px 0; }
.mode-section { margin: 32px 0; padding: 16px; background: white; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
.mode-section h2 { margin-top: 0; }
.mode-desc { color: #555; font-size: 14px; max-width: 800px; line-height: 1.5; }
.step { margin: 24px 0; }
.step h3 { margin: 0 0 4px 0; font-weight: 500; color: #444; }
.timeline-wrap { overflow-x: auto; padding-bottom: 6px; }
.axis-row { display: flex; align-items: center; margin: 4px 0; }
.track-row { display: flex; align-items: center; margin: 4px 0; }
.track-label {
  width: 60px; padding: 4px 8px; font-size: 12px; color: #444;
  border-left: 4px solid #ccc; box-sizing: border-box; flex-shrink: 0;
}
.track { position: relative; background: #f0f0f0; border: 1px solid #ddd; box-sizing: border-box; }
.axis { position: relative; height: 18px; }
.tick {
  position: absolute; top: 0; font-size: 10px; color: #666;
  border-left: 1px solid #bbb; padding-left: 2px; height: 100%; box-sizing: border-box;
}
.span {
  position: absolute; top: 2px; border-radius: 2px; cursor: pointer;
  box-sizing: border-box; opacity: 0.92;
}
.span:hover { opacity: 1; box-shadow: 0 0 0 1.5px #111; z-index: 100; }
.span-label {
  display: block; padding: 4px 6px; color: white; font-size: 11px;
  font-family: ui-monospace, monospace; white-space: nowrap;
  text-shadow: 0 1px 2px rgba(0,0,0,0.5); pointer-events: none;
}
.span-tables { margin-top: 8px; }
.span-tables summary { cursor: pointer; font-size: 12px; color: #555; user-select: none; padding: 4px 0; }
.tables-row { display: flex; gap: 24px; flex-wrap: wrap; margin-top: 6px; }
.span-table { border-collapse: collapse; font-size: 12px; font-family: ui-monospace, monospace; }
.span-table caption { caption-side: top; font-weight: 600; text-align: left; padding: 4px 0; color: #444; font-family: -apple-system, sans-serif; }
.span-table th, .span-table td { padding: 3px 8px; border-bottom: 1px solid #eee; text-align: left; }
.span-table th { background: #f8f8f8; font-weight: 600; }
.span-table td.num, .span-table th:nth-child(n+2) { text-align: right; font-variant-numeric: tabular-nums; }
.swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; vertical-align: middle; }
.summary tr:hover { background: #f8f8f8; }
.mode-summary { margin-top: 16px; }
.mode-summary summary { cursor: pointer; font-size: 13px; font-weight: 600; color: #333; padding: 6px 0; }
.step-summary {
  background: #f5f6f8; border: 1px solid #e0e2e6; border-radius: 6px;
  padding: 10px 14px; margin: 8px 0 12px; display: flex; flex-wrap: wrap;
  gap: 6px 18px; align-items: center; font-size: 13px;
}
.step-total { font-size: 14px; color: #222; margin-right: 12px; }
.pool-summary { display: flex; gap: 6px; align-items: center; }
.pool-name { font-weight: 600; color: #444; min-width: 50px; }
.pool-summary .total { color: #444; font-variant-numeric: tabular-nums; }
.cat-pill {
  display: inline-block; padding: 2px 8px; border-radius: 10px;
  color: white; font-size: 11px; font-variant-numeric: tabular-nums;
}
.bench-section { margin: 32px 0; }
.bench-section > h1 { background: #222; color: white; padding: 12px 16px; border-radius: 8px; margin: 0 0 12px 0; }
"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument(
        "--px-per-ms",
        type=float,
        default=2.5,
        help="Horizontal scale in pixels per millisecond. Bump higher for more label room.",
    )
    args = p.parse_args()

    # Bench prefix is the part of the filename before the first underscore:
    # wider_sync_poola_rank0.json → "wider"
    by_bench: dict[str, dict[str, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    for path in sorted(args.in_dir.glob("*_*_pool*_rank*.json")):
        data = json.loads(path.read_text())
        bench = path.name.split("_", 1)[0]
        by_bench[bench][data["mode"]][data["pool"]] = data

    bench_sections: list[str] = []
    for bench in sorted(by_bench.keys()):
        mode_sections: list[str] = []
        for mode in ("sync", "async"):
            if mode not in by_bench[bench]:
                continue
            pools = by_bench[bench][mode]
            if "a" not in pools or "b" not in pools:
                print(
                    f"[warn] bench={bench} mode={mode} missing pool data — got {list(pools.keys())}"
                )
                continue
            mode_sections.append(
                render_mode_section(
                    mode=mode,
                    pool_a=pools["a"],
                    pool_b=pools["b"],
                    px_per_ms=args.px_per_ms,
                )
            )
        if mode_sections:
            bench_sections.append(
                f'<div class="bench-section"><h1>{bench}</h1>{"".join(mode_sections)}</div>'
            )

    html = (
        '<!doctype html><html><head><meta charset="utf-8">'
        "<title>2-pool profile</title>"
        f"<style>{CSS}</style></head><body>"
        "<h1>2-pool — phase timeline</h1>"
        '<p style="color:#666;max-width:840px;line-height:1.5;">'
        "One section per benchmark; within each, sync and async profiler modes. "
        "Each step shows pool A (top track) and pool B (bottom track) sharing a "
        "common time origin (dist.barrier at step start). The colored stripe at "
        "the bottom of each box marks its category: "
        '<span class="cat-pill" style="background:#2ca02c">compute</span> '
        '<span class="cat-pill" style="background:#d62728">comm</span> '
        '<span class="cat-pill" style="background:#7f7f7f">wait</span>. '
        "The headline box above each timeline shows total step wall-clock and the "
        'compute/comm/wait split per pool. Click "raw spans" for exact numbers; '
        "hover any box for a precise tooltip."
        "</p>"
        f"{''.join(bench_sections)}"
        "</body></html>"
    )
    args.out.write_text(html)
    print(f"wrote {args.out} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
