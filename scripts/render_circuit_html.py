"""Render a circuit JSON file as a self-contained HTML page.

This is a thin wrapper that copies the circuit.html template and embeds the JSON
data inline, so the result is a single self-contained HTML file (no separate data file needed).

Usage:
    python scripts/render_circuit_html.py data/king_circuit.json -o circuit_standalone.html

Or from Python:
    from scripts.render_circuit_html import render_circuit_html
    render_circuit_html(Path("data/king_circuit.json"), Path("circuit_standalone.html"))
"""

import argparse
import json
from pathlib import Path


def render_circuit_html(json_path: Path, output_path: Path, title: str = "Circuit Graph") -> None:
    """Render a circuit JSON as a self-contained HTML file.

    Takes the interactive circuit.html template and replaces the fetch() call
    with inline data, producing a single portable HTML file.
    """
    with open(json_path) as f:
        data = json.load(f)

    # Read the template
    template_path = Path(__file__).parent.parent / "scripts" / "_circuit_template.html"
    if not template_path.exists():
        # Fallback: read from www
        from spd.settings import SPD_OUT_DIR

        template_path = SPD_OUT_DIR / "www" / "pile-editing" / "circuit.html"

    assert template_path.exists(), f"Template not found: {template_path}"
    template = template_path.read_text()

    # Replace the fetch with inline data
    inline_js = f"data = {json.dumps(data)}; init();"
    template = template.replace(
        "fetch(DATA_URL)\n"
        "  .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })\n"
        "  .then(d => { data = d; init(); })\n"
        "  .catch(e => { document.getElementById('stats').textContent = `Error: ${e.message}`; });",
        inline_js,
    )

    # Update title
    template = template.replace(
        '<title>Circuit Graph — King → "he"</title>', f"<title>{title}</title>"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(template)

    print(f"Wrote self-contained HTML to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render circuit JSON as self-contained HTML")
    parser.add_argument("json_path", type=Path, help="Path to circuit JSON file")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("circuit.html"), help="Output HTML path"
    )
    parser.add_argument("--title", default="Circuit Graph", help="Page title")
    args = parser.parse_args()
    render_circuit_html(args.json_path, args.output, args.title)
