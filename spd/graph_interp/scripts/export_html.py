"""Export graph interpretation data to JSON for the static HTML page.

Usage:
    python -m spd.graph_interp.scripts.export_html s-17805b61
    python -m spd.graph_interp.scripts.export_html s-17805b61 --subrun_id ti-20260223_213443
    python -m spd.graph_interp.scripts.export_html s-17805b61 --mock
"""

import json
import random
from dataclasses import asdict
from typing import Any

from spd.graph_interp.repo import GraphInterpRepo
from spd.graph_interp.schemas import LabelResult, get_graph_interp_dir
from spd.settings import SPD_OUT_DIR

WWW_DIR = SPD_OUT_DIR / "www"
DATA_DIR = WWW_DIR / "data"


def _label_to_dict(label: LabelResult) -> dict[str, str]:
    return {
        "label": label.label,
        "confidence": label.confidence,
        "reasoning": label.reasoning,
    }


def _parse_component_key(key: str) -> tuple[str, int]:
    layer, idx_str = key.rsplit(":", 1)
    return layer, int(idx_str)


def export_from_repo(repo: GraphInterpRepo) -> dict[str, Any]:
    output_labels = repo.get_all_output_labels()
    input_labels = repo.get_all_input_labels()
    unified_labels = repo.get_all_unified_labels()

    all_keys = sorted(
        set(output_labels) | set(input_labels) | set(unified_labels),
        key=lambda k: (_parse_component_key(k)[0], _parse_component_key(k)[1]),
    )

    components = []
    for key in all_keys:
        layer, component_idx = _parse_component_key(key)
        entry: dict[str, Any] = {
            "key": key,
            "layer": layer,
            "component_idx": component_idx,
        }
        if key in output_labels:
            entry["output_label"] = _label_to_dict(output_labels[key])
        if key in input_labels:
            entry["input_label"] = _label_to_dict(input_labels[key])
        if key in unified_labels:
            entry["unified_label"] = _label_to_dict(unified_labels[key])

        edges = repo.get_prompt_edges(key)
        if edges:
            entry["edges"] = [asdict(e) for e in edges]

        components.append(entry)

    label_counts = repo.get_label_counts()

    return {
        "decomposition_id": repo.run_id,
        "subrun_id": repo.subrun_id,
        "label_counts": label_counts,
        "components": components,
    }


def generate_mock_data(decomposition_id: str) -> dict[str, Any]:
    random.seed(42)

    layers = [
        "h.0.mlp.c_fc",
        "h.0.mlp.down_proj",
        "h.0.attn.q_proj",
        "h.0.attn.k_proj",
        "h.0.attn.v_proj",
        "h.0.attn.o_proj",
        "h.1.mlp.c_fc",
        "h.1.mlp.down_proj",
        "h.1.attn.q_proj",
        "h.1.attn.k_proj",
        "h.1.attn.v_proj",
        "h.1.attn.o_proj",
    ]

    output_labels_pool = [
        "sentence-final punctuation and period prediction",
        "proper nouns and character name completions",
        "emotional adjectives describing characters",
        "temporal adverbs and time-related transitions",
        "morphological suffix completion (-ing, -ed, -ly)",
        "determiners preceding concrete nouns",
        "dialogue-opening quotation marks and speech verbs",
        "plural noun suffixes after quantity words",
        "conjunction and clause boundary detection",
        "verb tense agreement and auxiliary verbs",
        "spatial prepositions and location descriptors",
        "possessive pronouns and genitive markers",
        "narrative action verbs (walked, looked, said)",
        "abstract emotion nouns (fear, joy, anger)",
        "comparative and superlative adjective forms",
    ]

    input_labels_pool = [
        "punctuation and common function words",
        "sentence-initial capital letters and proper nouns",
        "mid-sentence verbs following subject nouns",
        "adjective-noun boundaries in descriptive phrases",
        "clause-final positions before conjunctions",
        "article-noun sequences in noun phrases",
        "subject pronouns at clause boundaries",
        "preposition-object sequences",
        "verb stems preceding inflectional suffixes",
        "quotation marks and dialogue boundaries",
        "comma-separated list items",
        "sentence-medial adverbs after auxiliaries",
        "concrete nouns following determiners",
        "coordinating conjunctions between clauses",
        "word stems requiring morphological completion",
    ]

    unified_labels_pool = [
        "sentence termination tracking and terminal punctuation prediction",
        "character name recognition and proper noun completion",
        "emotional state description through adjective selection",
        "temporal transition signaling via adverbs and tense markers",
        "morphological word completion from stems to suffixed forms",
        "noun phrase construction: determiners predicting concrete nouns",
        "dialogue framing through quotation marks and speech attribution",
        "plural morphology following quantifiers and numerals",
        "clause coordination and syntactic boundary marking",
        "verbal agreement and auxiliary verb selection",
        "spatial relationship encoding via prepositional phrases",
        "possessive construction and genitive case marking",
        "narrative action sequencing through core verbs",
        "abstract emotional vocabulary and sentiment expression",
        "degree modification and comparative construction",
    ]

    confidences = ["high", "high", "high", "medium", "medium", "low"]

    reasoning_templates = [
        "The output function focuses on {output_focus}, while the input function responds to {input_focus}. Together, this component acts as a bridge between {bridge_from} and {bridge_to}, consistent with its position in {layer}.",
        "This component's output pattern of {output_focus} is activated by {input_focus} in the input. The unified interpretation captures how {bridge_from} contexts trigger {bridge_to} predictions.",
        "Downstream context shows this component feeds into {output_focus} pathways, while upstream context reveals activation by {input_focus}. The synthesis reflects a coherent role in {bridge_from}-to-{bridge_to} processing.",
    ]

    focus_terms = [
        "punctuation patterns",
        "noun completions",
        "verb inflections",
        "emotional descriptors",
        "syntactic boundaries",
        "morphological suffixes",
        "dialogue markers",
        "temporal signals",
        "spatial relationships",
    ]

    components = []
    for layer in layers:
        n_components = random.randint(8, 20)
        indices = sorted(random.sample(range(500), n_components))
        for idx in indices:
            key = f"{layer}:{idx}"
            conf = random.choice(confidences)
            output_conf = random.choice(confidences)
            input_conf = random.choice(confidences)

            output_label = random.choice(output_labels_pool)
            input_label = random.choice(input_labels_pool)
            unified_label = random.choice(unified_labels_pool)

            reasoning = random.choice(reasoning_templates).format(
                output_focus=random.choice(focus_terms),
                input_focus=random.choice(focus_terms),
                bridge_from=random.choice(focus_terms),
                bridge_to=random.choice(focus_terms),
                layer=layer,
            )

            components.append(
                {
                    "key": key,
                    "layer": layer,
                    "component_idx": idx,
                    "output_label": {
                        "label": output_label,
                        "confidence": output_conf,
                        "reasoning": f"Output: {reasoning}",
                    },
                    "input_label": {
                        "label": input_label,
                        "confidence": input_conf,
                        "reasoning": f"Input: {reasoning}",
                    },
                    "unified_label": {
                        "label": unified_label,
                        "confidence": conf,
                        "reasoning": reasoning,
                    },
                }
            )

    return {
        "decomposition_id": decomposition_id,
        "subrun_id": "ti-mock",
        "label_counts": {
            "output": len(components),
            "input": len(components),
            "unified": len(components),
        },
        "components": components,
    }


def main(
    decomposition_id: str,
    subrun_id: str | None = None,
    mock: bool = False,
) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / f"graph_interp_{decomposition_id}.json"

    if mock:
        data = generate_mock_data(decomposition_id)
        print(f"Generated mock data: {len(data['components'])} components")
    else:
        if subrun_id is not None:
            base_dir = get_graph_interp_dir(decomposition_id)
            subrun_dir = base_dir / subrun_id
            assert subrun_dir.exists(), f"Subrun dir not found: {subrun_dir}"
            db_path = subrun_dir / "interp.db"
            assert db_path.exists(), f"No interp.db in {subrun_dir}"
            from spd.graph_interp.db import GraphInterpDB

            db = GraphInterpDB(db_path, readonly=True)
            repo = GraphInterpRepo(db=db, subrun_dir=subrun_dir, run_id=decomposition_id)
        else:
            repo = GraphInterpRepo.open(decomposition_id)
            if repo is None:
                print(f"No graph interp data for {decomposition_id}. Generating mock data instead.")
                data = generate_mock_data(decomposition_id)
                with open(out_path, "w") as f:
                    json.dump(data, f)
                print(f"Wrote mock data to {out_path}")
                return

        data = export_from_repo(repo)
        print(f"Exported {len(data['components'])} components from {data['subrun_id']}")

    with open(out_path, "w") as f:
        json.dump(data, f)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
