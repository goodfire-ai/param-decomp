"""Generate dashboard JSON from harvest + autointerp data.

Outputs:
  - manifest.json: lightweight metadata + component list (inline in HTML)
  - components/{component_key}.json: per-component full data (loaded on demand)

Usage:
    python -m spd.paper_vis.generate --decomposition_id s-55ea3f9b --method vpd --out_dir out/vpd
"""

from pathlib import Path

import fire
import orjson

from spd.adapters import adapter_from_id
from spd.app.backend.app_tokenizer import AppTokenizer
from spd.autointerp.repo import InterpRepo
from spd.harvest.repo import HarvestRepo
from spd.harvest.schemas import ComponentData
from spd.paper_vis.data import (
    ActivationExampleData,
    ComponentDashboardData,
    DecompositionData,
    ScoreData,
    TokenPMIData,
    TokenSpan,
)


def _convert_pmi(pmi_data: list[tuple[int, float]], tok: AppTokenizer) -> list[tuple[str, float]]:
    return [(tok.get_tok_display(tid), score) for tid, score in pmi_data]


def _convert_example(example, tok: AppTokenizer, act_type: str) -> ActivationExampleData:
    activations = example.activations.get(act_type, [0.0] * len(example.token_ids))
    tokens = [
        TokenSpan(
            token=tok.get_tok_display(tid),
            is_firing=f,
            activation=a,
        )
        for tid, f, a in zip(example.token_ids, example.firings, activations, strict=True)
    ]
    center_idx = len(tokens) // 2
    return ActivationExampleData(tokens=tokens, center_idx=center_idx)


def _primary_activation_type(component: ComponentData) -> str:
    if not component.activation_examples:
        return "activation"
    act_types = list(component.activation_examples[0].activations.keys())
    if "causal_importance" in act_types:
        return "causal_importance"
    return act_types[0] if act_types else "activation"


def _build_component(
    comp: ComponentData,
    tok: AppTokenizer,
    interp_repo: InterpRepo | None,
    detection_scores: dict[str, float],
    fuzzing_scores: dict[str, float],
) -> ComponentDashboardData:
    act_type = _primary_activation_type(comp)
    mean_act = comp.mean_activations.get(act_type, 0.0)

    examples = [_convert_example(ex, tok, act_type) for ex in comp.activation_examples[:20]]

    input_pmi = TokenPMIData(
        top=_convert_pmi(comp.input_token_pmi.top, tok),
        bottom=_convert_pmi(comp.input_token_pmi.bottom, tok),
    )
    output_pmi = TokenPMIData(
        top=_convert_pmi(comp.output_token_pmi.top, tok),
        bottom=_convert_pmi(comp.output_token_pmi.bottom, tok),
    )

    label = None
    confidence = None
    reasoning = None
    detection_score = None
    fuzzing_score = None

    if interp_repo is not None:
        interp = interp_repo.get_interpretation(comp.component_key)
        if interp is not None:
            label = interp.label
            confidence = interp.confidence
            reasoning = interp.reasoning

        det_val = detection_scores.get(comp.component_key)
        if det_val is not None:
            detection_score = ScoreData(score=det_val, n_trials=0)

        fuz_val = fuzzing_scores.get(comp.component_key)
        if fuz_val is not None:
            fuzzing_score = ScoreData(score=fuz_val, n_trials=0)

    return ComponentDashboardData(
        component_key=comp.component_key,
        layer=comp.layer,
        component_idx=comp.component_idx,
        firing_density=comp.firing_density,
        mean_activation=mean_act,
        activation_examples=examples,
        input_token_pmi=input_pmi,
        output_token_pmi=output_pmi,
        label=label,
        confidence=confidence,
        reasoning=reasoning,
        detection_score=detection_score,
        fuzzing_score=fuzzing_score,
    )


def build_decomposition_data(
    decomposition_id: str,
    method: str,
    limit: int | None,
    out_dir: Path,
) -> DecompositionData:
    adapter = adapter_from_id(decomposition_id)

    harvest = HarvestRepo.open_most_recent(decomposition_id)
    assert harvest is not None, f"No harvest data for {decomposition_id}"

    tok = AppTokenizer.from_pretrained(adapter.tokenizer_name)

    interp_repo: InterpRepo | None = None
    try:
        interp_repo = InterpRepo.open(decomposition_id)
    except Exception:
        pass

    detection_scores: dict[str, float] = {}
    fuzzing_scores: dict[str, float] = {}
    if interp_repo is not None:
        detection_scores = interp_repo.get_scores("detection")
        fuzzing_scores = interp_repo.get_scores("fuzzing")

    summaries = harvest.get_summary()
    keys = list(summaries.keys())
    if limit is not None:
        keys = keys[:limit]

    comp_dir = out_dir / "components"
    comp_dir.mkdir(parents=True, exist_ok=True)

    dashboard_components: list[ComponentDashboardData] = []
    for i, key in enumerate(keys):
        comp = harvest.get_component(key)
        assert comp is not None
        dash_comp = _build_component(comp, tok, interp_repo, detection_scores, fuzzing_scores)
        dashboard_components.append(dash_comp)

        safe_key = key.replace(":", "_").replace("/", "_")
        comp_path = comp_dir / f"{safe_key}.json"
        comp_path.write_bytes(orjson.dumps(dash_comp.model_dump()))

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(keys)} components", flush=True)

    print(f"  {len(keys)}/{len(keys)} components done", flush=True)

    layers = adapter.layer_activation_sizes
    return DecompositionData(
        decomposition_id=decomposition_id,
        method=method,
        base_model=adapter.tokenizer_name,
        n_components=sum(n for _, n in layers),
        n_layers=len(layers),
        components=dashboard_components,
    )


def main(
    decomposition_id: str,
    method: str,
    out_dir: str,
    limit: int | None = None,
) -> None:
    out_path = Path(out_dir)
    data = build_decomposition_data(decomposition_id, method, limit, out_path)
    manifest = out_path / "manifest.json"
    manifest.write_bytes(orjson.dumps(data.model_dump(exclude={"components"}), option=orjson.OPT_INDENT_2))
    print(f"Wrote {len(data.components)} components to {out_path}/")


if __name__ == "__main__":
    fire.Fire(main)
