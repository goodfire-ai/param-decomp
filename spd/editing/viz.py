"""HTML visualization for component edit comparisons. Intended for Jupyter notebooks."""

import html as html_lib
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spd.editing.compare import ExampleDiff, TokenDiff


def render_edit_comparison(
    diffs: list["ExampleDiff"],
    title: str,
    subtitle: str,
) -> str:
    """Render per-token KL / CI / activation heatmap with hover tooltips.

    Returns an HTML string. Use IPython.display.HTML() to render in notebooks.
    """
    assert diffs, "No examples to render"

    max_kl = max(d.max_kl for d in diffs)
    max_ci = max(t.ci for d in diffs for t in d.tokens)
    max_act = max(abs(t.activation) for d in diffs for t in d.tokens)

    uid = f"ec{id(diffs)}"

    def bg_kl(kl: float) -> str:
        if kl < 0.01:
            return "transparent"
        i = min(1.0, math.log1p(kl) / math.log1p(max_kl))
        return f"rgba({int(255 * i)},{int(40 * (1 - i))},{int(40 * (1 - i))},{i})"

    def bg_ci(ci: float) -> str:
        if ci < 1e-4:
            return "transparent"
        i = min(1.0, ci / max(max_ci, 1e-6))
        return f"rgba({int(80 + 100 * i)},{int(160 + 80 * i)},255,{i})"

    def bg_act(act: float) -> str:
        if abs(act) < 1e-4:
            return "transparent"
        i = min(1.0, abs(act) / max(max_act, 1e-6))
        if act > 0:
            return f"rgba({int(50 + 150 * i)},{int(200 + 55 * i)},80,{i})"
        return f"rgba({int(200 + 55 * i)},{int(50 + 100 * i)},180,{i})"

    e = html_lib.escape
    parts: list[str] = []
    parts.append(_CSS.replace("UID", uid))
    parts.append(f'<div id="{uid}">')
    parts.append(f'<div class="title">{e(title)}</div>')
    parts.append(f'<div class="subtitle">{e(subtitle)}</div>')

    # Mode switcher
    parts.append('<div class="controls">')
    for mode, label in [
        ("kl", "KL Divergence"),
        ("ci", "Causal Importance"),
        ("act", "Component Activation"),
    ]:
        checked = "checked" if mode == "kl" else ""
        parts.append(
            f'<input type="radio" name="{uid}_m" id="{uid}_{mode}" value="{mode}" {checked}>'
        )
        parts.append(f'<label for="{uid}_{mode}">{label}</label>')
    parts.append("</div>")

    for ex_idx, diff in enumerate(diffs):
        parts.append(
            f'<div class="ex"><div class="meta">KL {diff.max_kl:.1f}</div><div class="toks">'
        )

        for t in diff.tokens:
            cls = "t fire" if t.fires else "t"
            parts.append(
                f'<span class="{cls}" '
                f'data-kl="{bg_kl(t.kl)}" data-ci="{bg_ci(t.ci)}" data-act="{bg_act(t.activation)}" '
                f'style="background:{bg_kl(t.kl)};">'
            )
            parts.append(_tooltip(t))
            parts.append(f"{e(t.span)}</span>")

        parts.append("</div></div>")
        if ex_idx < len(diffs) - 1:
            parts.append('<div class="sep"></div>')

    parts.append("</div>")
    parts.append(_JS.replace("UID", uid))
    return "\n".join(parts)


def _tooltip(t: "TokenDiff") -> str:
    """Build the hover tooltip HTML for a single token."""
    esc = html_lib.escape  # local ref for brevity
    p: list[str] = ['<span class="tip">']
    p.append(
        f'<div style="margin-bottom:6px"><span class="lbl">KL:</span> {t.kl:.4f}'
        f' &nbsp; <span class="lbl">CI:</span> {t.ci:.4f}'
        f' &nbsp; <span class="lbl">Act:</span> {t.activation:.4f}</div>'
    )

    # Before / After columns
    p.append('<div class="cols"><div class="col">')
    p.append('<div class="col-title">Before (top-k)</div><table>')
    for tok_str, prob in t.topk_before:
        p.append(f"<tr><td>{esc(tok_str)}</td><td>{prob:.1%}</td></tr>")
    p.append('</table></div><div class="col">')
    p.append('<div class="col-title">After (top-k)</div><table>')
    for tok_str, prob in t.topk_after:
        p.append(f"<tr><td>{esc(tok_str)}</td><td>{prob:.1%}</td></tr>")
    p.append("</table></div></div>")

    # Increase / Decrease columns
    p.append('<div style="margin-top:6px;border-top:1px solid #30363d;padding-top:4px">')
    p.append('<div class="cols"><div class="col">')
    p.append('<div class="col-title">▲ Top increases</div><table>')
    for tok_str, bp, ep in t.top_increases:
        p.append(
            f'<tr class="inc"><td>{esc(tok_str)}</td><td>{bp:.1%}</td><td>→</td><td>{ep:.1%}</td></tr>'
        )
    p.append('</table></div><div class="col">')
    p.append('<div class="col-title">▼ Top decreases</div><table>')
    for tok_str, bp, ep in t.top_decreases:
        p.append(
            f'<tr class="dec"><td>{esc(tok_str)}</td><td>{bp:.1%}</td><td>→</td><td>{ep:.1%}</td></tr>'
        )
    p.append("</table></div></div></div>")

    p.append("</span>")
    return "\n".join(p)


_CSS = """<style>
#UID { font-family: 'Menlo','Monaco',monospace; font-size: 12px; color: #c9d1d9; background: #0d1117; padding: 16px; border-radius: 8px; }
#UID .title { font-size: 14px; font-weight: bold; color: #e6edf3; margin-bottom: 4px; }
#UID .subtitle { font-size: 11px; color: #8b949e; margin-bottom: 12px; }
#UID .controls { display: flex; gap: 12px; margin-bottom: 14px; }
#UID .controls label { font-size: 11px; color: #8b949e; cursor: pointer; padding: 3px 8px; border: 1px solid #30363d; border-radius: 4px; }
#UID .controls input:checked+label { background: #21262d; color: #e6edf3; border-color: #58a6ff; }
#UID .controls input { display: none; }
#UID .ex { margin-bottom: 4px; display: flex; align-items: flex-start; gap: 8px; }
#UID .meta { min-width: 70px; font-size: 10px; color: #8b949e; padding-top: 3px; text-align: right; flex-shrink: 0; }
#UID .toks { display: flex; flex-wrap: wrap; align-items: baseline; }
#UID .t { padding: 0 1px; border-radius: 2px; cursor: pointer; position: relative; border-bottom: 2px solid transparent; line-height: 1.3; transition: background .15s; white-space: pre; }
#UID .t.fire { border-bottom: 2px solid #f0883e; }
#UID .tip { display: none; position: absolute; bottom: calc(100% + 4px); left: 50%; transform: translateX(-50%); background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 10px; font-size: 10px; white-space: nowrap; z-index: 100; min-width: 340px; box-shadow: 0 4px 12px rgba(0,0,0,.5); }
#UID .t:hover .tip { display: block; }
#UID .tip .inc { color: #7ee787; }
#UID .tip .dec { color: #f85149; }
#UID .tip .lbl { color: #8b949e; font-size: 9px; }
#UID .tip table { border-collapse: collapse; width: 100%; }
#UID .tip td { padding: 1px 3px; }
#UID .tip .cols { display: flex; gap: 12px; }
#UID .tip .col { flex: 1; }
#UID .tip .col-title { color: #58a6ff; font-size: 9px; font-weight: bold; margin-bottom: 2px; }
#UID .sep { height: 1px; background: #21262d; margin: 6px 0; }
</style>"""

_JS = """<script>
(function(){
  var c=document.getElementById('UID');
  c.querySelectorAll('input[name="UID_m"]').forEach(function(r){
    r.addEventListener('change',function(){
      var m=this.value;
      c.querySelectorAll('.t').forEach(function(t){t.style.background=t.getAttribute('data-'+m)});
    });
  });
})();
</script>"""
