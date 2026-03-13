"""Build a dashboard with per-component JSON files for incremental loading.

Outputs:
  <out_dir>/
    index.html          — self-contained dashboard shell
    vpd/manifest.json   — VPD metadata (no heavy component data)
    vpd/components/     — per-component JSON files
    tc/manifest.json    — transcoder metadata
    tc/components/      — per-component JSON files

Usage:
    python -m spd.paper_vis.build_dashboard \
        --vpd_id s-55ea3f9b --tc_id tc-3f297233 \
        --out_dir dashboard_out --limit 50
"""

from pathlib import Path

import fire
import orjson

from spd.paper_vis.generate import build_decomposition_data


def _component_index(data) -> list[dict]:
    """Extract lightweight component index from full data (for inline embedding)."""
    return [
        {
            "component_key": c.component_key,
            "layer": c.layer,
            "component_idx": c.component_idx,
            "firing_density": c.firing_density,
            "mean_activation": c.mean_activation,
            "label": c.label,
            "confidence": c.confidence,
            "detection_score": c.detection_score.model_dump() if c.detection_score else None,
            "fuzzing_score": c.fuzzing_score.model_dump() if c.fuzzing_score else None,
        }
        for c in data.components
    ]


def build(
    out_dir: str = "dashboard_out",
    vpd_id: str | None = None,
    tc_id: str | None = None,
    limit: int | None = None,
) -> None:
    assert vpd_id or tc_id, "Provide at least one of --vpd_id or --tc_id"

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {"vpd": None, "transcoder": None}

    if vpd_id:
        print(f"Loading VPD data: {vpd_id}")
        vpd_data = build_decomposition_data(vpd_id, "vpd", limit, out / "vpd")
        manifest["vpd"] = {
            **vpd_data.model_dump(exclude={"components"}),
            "component_index": _component_index(vpd_data),
            "components_path": "vpd/components",
        }

    if tc_id:
        print(f"Loading transcoder data: {tc_id}")
        tc_data = build_decomposition_data(tc_id, "transcoder", limit, out / "tc")
        manifest["transcoder"] = {
            **tc_data.model_dump(exclude={"components"}),
            "component_index": _component_index(tc_data),
            "components_path": "tc/components",
        }

    template_path = Path(__file__).parent / "dashboard.html"
    html = template_path.read_text()
    manifest_json = orjson.dumps(manifest).decode()
    html = html.replace("/*DATA_JSON*/null", manifest_json)

    (out / "index.html").write_text(html)
    print(f"Wrote dashboard to {out}/index.html")


if __name__ == "__main__":
    fire.Fire(build)
