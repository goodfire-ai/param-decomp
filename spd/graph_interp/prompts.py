"""Prompt formatters for graph interpretation.

Three prompts:
1. Output pass (late→early): "What does this component DO?" — output tokens, says examples, downstream
2. Input pass (early→late): "What TRIGGERS this component?" — input tokens, fires-on examples, upstream
3. Unification: Synthesize output + input labels into unified label.
"""

from spd.app.backend.app_tokenizer import AppTokenizer
from spd.autointerp.prompt_helpers import (
    build_fires_on_examples,
    build_input_section,
    build_output_section,
    build_says_examples,
    density_note,
    human_layer_desc,
    layer_position_note,
)
from spd.autointerp.schemas import ModelMetadata
from spd.graph_interp.graph_context import RelatedComponent
from spd.graph_interp.schemas import LabelResult
from spd.harvest.analysis import TokenPRLift
from spd.harvest.schemas import ComponentData

LABEL_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "label": {"type": "string"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "reasoning": {"type": "string"},
    },
    "required": ["label", "confidence", "reasoning"],
    "additionalProperties": False,
}


def _component_header(
    component: ComponentData,
    model_metadata: ModelMetadata,
) -> str:
    canonical = model_metadata.layer_descriptions.get(component.layer, component.layer)
    layer_desc = human_layer_desc(canonical, model_metadata.n_blocks)
    position_note = layer_position_note(canonical, model_metadata.n_blocks)
    dens_note = density_note(component.firing_density)

    rate_str = (
        f"~1 in {int(1 / component.firing_density)} tokens"
        if component.firing_density > 0.0
        else "extremely rare"
    )

    context_notes = " ".join(filter(None, [position_note, dens_note]))

    return f"""\
## Context
- Component: {layer_desc} (component {component.component_idx}), {model_metadata.n_blocks}-block model
- Firing rate: {component.firing_density * 100:.2f}% ({rate_str})
{context_notes}"""


def format_output_prompt(
    component: ComponentData,
    model_metadata: ModelMetadata,
    app_tok: AppTokenizer,
    output_token_stats: TokenPRLift,
    related: list[RelatedComponent],
    label_max_words: int,
    max_examples: int,
) -> str:
    header = _component_header(component, model_metadata)

    output_pmi = (
        [(app_tok.get_tok_display(tid), pmi) for tid, pmi in component.output_token_pmi.top]
        if component.output_token_pmi.top
        else None
    )
    output_section = build_output_section(output_token_stats, output_pmi)
    says = build_says_examples(component, app_tok, max_examples)
    related_table = _format_related_table(related, model_metadata, app_tok)

    return f"""\
You are analyzing a component in a neural network to understand its OUTPUT FUNCTION — what it does when it fires.

{header}

## Output tokens (what the model produces when this component fires)
{output_section}
## Activation examples — what the model produces
{says}
## Downstream components (what this component influences)
These components in later layers are most influenced by this component (by gradient attribution):
{related_table}
## Task
Give a {label_max_words}-word-or-fewer label describing this component's OUTPUT FUNCTION — what it does when it fires.

Say "unclear" if the evidence is too weak.

Respond with JSON: {{"label": "...", "confidence": "low|medium|high", "reasoning": "..."}}
"""


def format_input_prompt(
    component: ComponentData,
    model_metadata: ModelMetadata,
    app_tok: AppTokenizer,
    input_token_stats: TokenPRLift,
    related: list[RelatedComponent],
    label_max_words: int,
    max_examples: int,
) -> str:
    header = _component_header(component, model_metadata)

    input_pmi = (
        [(app_tok.get_tok_display(tid), pmi) for tid, pmi in component.input_token_pmi.top]
        if component.input_token_pmi.top
        else None
    )
    input_section = build_input_section(input_token_stats, input_pmi)
    fires_on = build_fires_on_examples(component, app_tok, max_examples)
    related_table = _format_related_table(related, model_metadata, app_tok)

    return f"""\
You are analyzing a component in a neural network to understand its INPUT FUNCTION — what triggers it to fire.

{header}

## Input tokens (what causes this component to fire)
{input_section}
## Activation examples — where the component fires
{fires_on}
## Upstream components (what feeds into this component)
These components in earlier layers most strongly attribute to this component:
{related_table}
## Task
Give a {label_max_words}-word-or-fewer label describing this component's INPUT FUNCTION — what conditions trigger it to fire.

Say "unclear" if the evidence is too weak.

Respond with JSON: {{"label": "...", "confidence": "low|medium|high", "reasoning": "..."}}
"""


def format_unification_prompt(
    output_label: LabelResult,
    input_label: LabelResult,
    component: ComponentData,
    model_metadata: ModelMetadata,
    app_tok: AppTokenizer,
    label_max_words: int,
    max_examples: int,
) -> str:
    header = _component_header(component, model_metadata)
    fires_on = build_fires_on_examples(component, app_tok, max_examples)
    says = build_says_examples(component, app_tok, max_examples)

    return f"""\
A neural network component has been analyzed from two perspectives.

{header}

## Activation examples — where the component fires
{fires_on}
## Activation examples — what the model produces
{says}
## Two-perspective analysis

OUTPUT FUNCTION: "{output_label.label}" (confidence: {output_label.confidence})
  Reasoning: {output_label.reasoning}

INPUT FUNCTION: "{input_label.label}" (confidence: {input_label.confidence})
  Reasoning: {input_label.reasoning}

## Task
Synthesize these into a single unified label (max {label_max_words} words) that captures the component's complete role. If input and output suggest the same concept, unify them. If they describe genuinely different aspects (e.g. fires on X, produces Y), combine both.

Respond with JSON: {{"label": "...", "confidence": "low|medium|high", "reasoning": "..."}}
"""


def _format_related_table(
    components: list[RelatedComponent],
    model_metadata: ModelMetadata,
    app_tok: AppTokenizer,
) -> str:
    # Filter: only show labeled components and token entries (embed/output)
    visible = [n for n in components if n.label is not None or _is_token_entry(n.component_key)]
    if not visible:
        return "(no related components with labels found)\n"

    # Normalize attributions: strongest = 1.0
    max_attr = max(abs(n.attribution) for n in visible)
    norm = max_attr if max_attr > 0 else 1.0

    lines: list[str] = []
    for n in visible:
        display = _component_display(n.component_key, model_metadata, app_tok)
        rel_attr = n.attribution / norm

        parts = [f"  {display} (relative attribution: {rel_attr:+.2f}"]
        if n.jaccard is not None:
            parts.append(f", co-firing Jaccard: {n.jaccard:.3f}")
        parts.append(")")

        line = "".join(parts)
        if n.label is not None:
            line += f'\n    label: "{n.label}" (confidence: {n.confidence})'
        lines.append(line)

    return "\n".join(lines) + "\n"


def _is_token_entry(key: str) -> bool:
    layer = key.rsplit(":", 1)[0]
    return layer in ("embed", "output")


def _component_display(key: str, model_metadata: ModelMetadata, app_tok: AppTokenizer) -> str:
    layer, idx_str = key.rsplit(":", 1)
    match layer:
        case "embed":
            return f'input token "{app_tok.get_tok_display(int(idx_str))}"'
        case "output":
            return f'output token "{app_tok.get_tok_display(int(idx_str))}"'
        case _:
            canonical = model_metadata.layer_descriptions.get(layer, layer)
            desc = human_layer_desc(canonical, model_metadata.n_blocks)
            return f"component from {desc}"
