"""Graph interpretation endpoints.

Serves context-aware component labels (output/input/unified) and the
prompt-edge graph produced by the graph_interp pipeline.
"""

import random

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from spd.app.backend.dependencies import DepLoadedRun
from spd.app.backend.utils import log_errors
from spd.graph_interp.schemas import LabelResult
from spd.topology import TransformerTopology

# TODO(oli): Remove MOCK_MODE after real data is available
MOCK_MODE = False

MAX_GRAPH_NODES = 500


_ALREADY_CANONICAL = {"embed", "output"}


def _concrete_to_canonical_key(concrete_key: str, topology: TransformerTopology) -> str:
    layer, idx = concrete_key.rsplit(":", 1)
    if layer in _ALREADY_CANONICAL:
        return concrete_key
    canonical = topology.target_to_canon(layer)
    return f"{canonical}:{idx}"


def _canonical_to_concrete_key(
    canonical_layer: str, component_idx: int, topology: TransformerTopology
) -> str:
    concrete = topology.canon_to_target(canonical_layer)
    return f"{concrete}:{component_idx}"


# -- Schemas -------------------------------------------------------------------


class GraphInterpHeadline(BaseModel):
    label: str
    confidence: str
    output_label: str | None
    input_label: str | None


class LabelDetail(BaseModel):
    label: str
    confidence: str
    reasoning: str
    prompt: str


class GraphInterpDetail(BaseModel):
    output: LabelDetail | None
    input: LabelDetail | None
    unified: LabelDetail | None


class PromptEdgeResponse(BaseModel):
    related_key: str
    pass_name: str
    attribution: float
    related_label: str | None
    related_confidence: str | None
    token_str: str | None


class GraphInterpComponentDetail(BaseModel):
    output: LabelDetail | None
    input: LabelDetail | None
    unified: LabelDetail | None
    edges: list[PromptEdgeResponse]


class GraphNode(BaseModel):
    component_key: str
    label: str
    confidence: str


class GraphEdge(BaseModel):
    source: str
    target: str
    attribution: float
    pass_name: str


class ModelGraphResponse(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]


# -- Router --------------------------------------------------------------------

router = APIRouter(prefix="/api/graph_interp", tags=["graph_interp"])


@router.get("/labels")
@log_errors
def get_all_labels(loaded: DepLoadedRun) -> dict[str, GraphInterpHeadline]:
    if MOCK_MODE:
        return _mock_all_labels(loaded)

    repo = loaded.graph_interp
    if repo is None:
        return {}

    topology = loaded.topology
    unified = repo.get_all_unified_labels()
    output = repo.get_all_output_labels()
    input_ = repo.get_all_input_labels()

    all_keys = set(unified) | set(output) | set(input_)
    result: dict[str, GraphInterpHeadline] = {}

    for concrete_key in all_keys:
        u = unified.get(concrete_key)
        o = output.get(concrete_key)
        i = input_.get(concrete_key)

        label = u or o or i
        assert label is not None
        canonical_key = _concrete_to_canonical_key(concrete_key, topology)

        result[canonical_key] = GraphInterpHeadline(
            label=label.label,
            confidence=label.confidence,
            output_label=o.label if o else None,
            input_label=i.label if i else None,
        )

    return result


def _to_detail(label: LabelResult | None) -> LabelDetail | None:
    if label is None:
        return None
    return LabelDetail(
        label=label.label,
        confidence=label.confidence,
        reasoning=label.reasoning,
        prompt=label.prompt,
    )


@router.get("/labels/{layer}/{c_idx}")
@log_errors
def get_label_detail(layer: str, c_idx: int, loaded: DepLoadedRun) -> GraphInterpDetail:
    if MOCK_MODE:
        return _mock_label_detail(layer, c_idx)

    repo = loaded.graph_interp
    if repo is None:
        raise HTTPException(status_code=404, detail="Graph interp data not available")

    concrete_key = _canonical_to_concrete_key(layer, c_idx, loaded.topology)

    return GraphInterpDetail(
        output=_to_detail(repo.get_output_label(concrete_key)),
        input=_to_detail(repo.get_input_label(concrete_key)),
        unified=_to_detail(repo.get_unified_label(concrete_key)),
    )


@router.get("/detail/{layer}/{c_idx}")
@log_errors
def get_component_detail(
    layer: str, c_idx: int, loaded: DepLoadedRun
) -> GraphInterpComponentDetail:
    repo = loaded.graph_interp
    if repo is None:
        raise HTTPException(status_code=404, detail="Graph interp data not available")

    topology = loaded.topology
    concrete_key = _canonical_to_concrete_key(layer, c_idx, topology)

    raw_edges = repo.get_prompt_edges(concrete_key)
    tokenizer = loaded.tokenizer
    edges = []
    for e in raw_edges:
        rel_layer, rel_idx = e.related_key.rsplit(":", 1)
        token_str = tokenizer.decode([int(rel_idx)]) if rel_layer in ("embed", "output") else None
        edges.append(
            PromptEdgeResponse(
                related_key=_concrete_to_canonical_key(e.related_key, topology),
                pass_name=e.pass_name,
                attribution=e.attribution,
                related_label=e.related_label,
                related_confidence=e.related_confidence,
                token_str=token_str,
            )
        )

    return GraphInterpComponentDetail(
        output=_to_detail(repo.get_output_label(concrete_key)),
        input=_to_detail(repo.get_input_label(concrete_key)),
        unified=_to_detail(repo.get_unified_label(concrete_key)),
        edges=edges,
    )


@router.get("/graph")
@log_errors
def get_model_graph(loaded: DepLoadedRun) -> ModelGraphResponse:
    if MOCK_MODE:
        return _mock_model_graph(loaded)

    repo = loaded.graph_interp
    if repo is None:
        raise HTTPException(status_code=404, detail="Graph interp data not available")

    topology = loaded.topology

    unified = repo.get_all_unified_labels()
    nodes = []
    for concrete_key, label in unified.items():
        canonical_key = _concrete_to_canonical_key(concrete_key, topology)
        nodes.append(
            GraphNode(
                component_key=canonical_key,
                label=label.label,
                confidence=label.confidence,
            )
        )

    nodes = nodes[:MAX_GRAPH_NODES]
    node_keys = {n.component_key for n in nodes}

    raw_edges = repo.get_all_prompt_edges()
    edges = []
    for e in raw_edges:
        comp_canon = _concrete_to_canonical_key(e.component_key, topology)
        rel_canon = _concrete_to_canonical_key(e.related_key, topology)

        match e.pass_name:
            case "output":
                source, target = comp_canon, rel_canon
            case "input":
                source, target = rel_canon, comp_canon

        if source not in node_keys or target not in node_keys:
            continue

        edges.append(
            GraphEdge(
                source=source,
                target=target,
                attribution=e.attribution,
                pass_name=e.pass_name,
            )
        )

    return ModelGraphResponse(nodes=nodes, edges=edges)


# -- Mock data (TODO: remove after real data available) ------------------------

_MOCK_LABELS = [
    "sentence-final punctuation",
    "proper noun completion",
    "emotional adjective selection",
    "temporal adverb prediction",
    "morphological suffix (-ing/-ed)",
    "determiner before noun",
    "dialogue quotation marks",
    "plural noun suffix",
    "clause boundary detection",
    "verb tense agreement",
    "spatial preposition",
    "possessive pronoun",
    "narrative action verb",
    "abstract emotion noun",
    "comparative adjective form",
    "subject-verb agreement",
    "article selection (a/the)",
    "comma splice detection",
    "pronoun resolution",
    "negation scope",
]

_MOCK_INPUT_LABELS = [
    "sentence-initial capitals",
    "mid-sentence verb position",
    "adjective-noun boundary",
    "clause-final position",
    "article-noun sequence",
    "subject pronoun at boundary",
    "preposition-object pair",
    "verb stem before suffix",
    "quotation boundary",
    "comma-separated items",
]


def _mock_all_labels(loaded: DepLoadedRun) -> dict[str, GraphInterpHeadline]:
    rng = random.Random(42)
    topology = loaded.topology
    confidences = ["high", "high", "high", "medium", "medium", "low"]

    result: dict[str, GraphInterpHeadline] = {}
    for target_path, components in loaded.model.components.items():
        canon = topology.target_to_canon(target_path)
        n_components = components.C
        n_mock = min(n_components, rng.randint(5, 20))
        indices = sorted(rng.sample(range(n_components), n_mock))
        for idx in indices:
            key = f"{canon}:{idx}"
            result[key] = GraphInterpHeadline(
                label=rng.choice(_MOCK_LABELS),
                confidence=rng.choice(confidences),
                output_label=rng.choice(_MOCK_LABELS),
                input_label=rng.choice(_MOCK_INPUT_LABELS),
            )
    return result


def _mock_label_detail(layer: str, c_idx: int) -> GraphInterpDetail:
    rng = random.Random(hash((layer, c_idx)))
    conf = rng.choice(["high", "medium", "low"])
    return GraphInterpDetail(
        output=LabelDetail(
            label=rng.choice(_MOCK_LABELS),
            confidence=conf,
            reasoning=f"Output: Component {layer}:{c_idx} writes {rng.choice(_MOCK_LABELS).lower()} tokens to the residual stream.",
            prompt="(mock prompt)",
        ),
        input=LabelDetail(
            label=rng.choice(_MOCK_INPUT_LABELS),
            confidence=conf,
            reasoning=f"Input: Component {layer}:{c_idx} fires on {rng.choice(_MOCK_INPUT_LABELS).lower()} patterns.",
            prompt="(mock prompt)",
        ),
        unified=LabelDetail(
            label=rng.choice(_MOCK_LABELS),
            confidence=conf,
            reasoning=f"Unified: Combines output ({rng.choice(_MOCK_LABELS).lower()}) and input ({rng.choice(_MOCK_INPUT_LABELS).lower()}) functions.",
            prompt="(mock prompt)",
        ),
    )


def _mock_model_graph(loaded: DepLoadedRun) -> ModelGraphResponse:
    labels = _mock_all_labels(loaded)

    nodes = [
        GraphNode(component_key=key, label=h.label, confidence=h.confidence)
        for key, h in labels.items()
    ]

    rng = random.Random(42)
    keys = list(labels.keys())
    edges: list[GraphEdge] = []

    for key in keys:
        layer = key.rsplit(":", 1)[0]
        later_keys = [k for k in keys if k.rsplit(":", 1)[0] != layer]
        n_edges = rng.randint(1, 4)
        for target in rng.sample(later_keys, min(n_edges, len(later_keys))):
            edges.append(
                GraphEdge(
                    source=key,
                    target=target,
                    attribution=rng.uniform(-1.0, 1.0),
                    pass_name=rng.choice(["output", "input"]),
                )
            )

    return ModelGraphResponse(nodes=nodes, edges=edges)
