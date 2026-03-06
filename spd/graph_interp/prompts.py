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
    token_pmi_pairs,
)
from spd.autointerp.schemas import ModelMetadata
from spd.graph_interp.graph_context import RelatedComponent
from spd.graph_interp.schemas import LabelResult
from spd.harvest.analysis import TokenPRLift
from spd.harvest.schemas import ComponentData
from spd.utils.markdown import Md

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

JSON_INSTRUCTION = (
    'Respond with JSON: {"label": "...", "confidence": "low|medium|high", "reasoning": "..."}'
)

UNCLEAR_NOTE = 'Say "unclear" if the evidence is too weak.'


def _component_header(
    component: ComponentData,
    model_metadata: ModelMetadata,
) -> Md:
    canonical = model_metadata.layer_descriptions.get(component.layer, component.layer)
    layer_desc = human_layer_desc(canonical, model_metadata.n_blocks)
    position_note = layer_position_note(canonical, model_metadata.n_blocks)
    dens_note = density_note(component.firing_density)

    rate_str = (
        f"~1 in {int(1 / component.firing_density)} tokens"
        if component.firing_density > 0.0
        else "extremely rare"
    )

    md = Md()
    md.h(2, "Context").bullets(
        [
            f"Component: {layer_desc} (component {component.component_idx}), {model_metadata.n_blocks}-block model",
            f"Firing rate: {component.firing_density * 100:.2f}% ({rate_str})",
        ]
    )
    context_notes = " ".join(filter(None, [position_note, dens_note]))
    if context_notes:
        md.p(context_notes)
    return md


def format_output_prompt(
    component: ComponentData,
    model_metadata: ModelMetadata,
    app_tok: AppTokenizer,
    output_token_stats: TokenPRLift,
    related: list[RelatedComponent],
    label_max_words: int,
    max_examples: int,
) -> str:
    output_pmi = token_pmi_pairs(app_tok, component.output_token_pmi.top)

    md = Md()
    md.p(
        "You are analyzing a component in a neural network to understand its "
        "OUTPUT FUNCTION — what it does when it fires."
    )
    md.extend(_component_header(component, model_metadata))

    md.h(2, "Output tokens (what the model produces when this component fires)")
    md.extend(build_output_section(output_token_stats, output_pmi))

    md.h(2, "Activation examples — what the model produces")
    md.extend(build_says_examples(component, app_tok, max_examples))

    md.h(2, "Downstream components (what this component influences)")
    md.p(
        "These components in later layers are most influenced by this component (by gradient attribution):"
    )
    md.extend(_format_related(related, model_metadata, app_tok))

    md.h(2, "Task")
    md.p(
        f"Give a {label_max_words}-word-or-fewer label describing this component's "
        "OUTPUT FUNCTION — what it does when it fires."
    )
    md.p(UNCLEAR_NOTE).p(JSON_INSTRUCTION)

    return md.build()


def format_input_prompt(
    component: ComponentData,
    model_metadata: ModelMetadata,
    app_tok: AppTokenizer,
    input_token_stats: TokenPRLift,
    related: list[RelatedComponent],
    label_max_words: int,
    max_examples: int,
) -> str:
    input_pmi = token_pmi_pairs(app_tok, component.input_token_pmi.top)

    md = Md()
    md.p(
        "You are analyzing a component in a neural network to understand its "
        "INPUT FUNCTION — what triggers it to fire."
    )
    md.extend(_component_header(component, model_metadata))

    md.h(2, "Input tokens (what causes this component to fire)")
    md.extend(build_input_section(input_token_stats, input_pmi))

    md.h(2, "Activation examples — where the component fires")
    md.extend(build_fires_on_examples(component, app_tok, max_examples))

    md.h(2, "Upstream components (what feeds into this component)")
    md.p("These components in earlier layers most strongly attribute to this component:")
    md.extend(_format_related(related, model_metadata, app_tok))

    md.h(2, "Task")
    md.p(
        f"Give a {label_max_words}-word-or-fewer label describing this component's "
        "INPUT FUNCTION — what conditions trigger it to fire."
    )
    md.p(UNCLEAR_NOTE).p(JSON_INSTRUCTION)

    return md.build()


def format_unification_prompt(
    output_label: LabelResult,
    input_label: LabelResult,
    component: ComponentData,
    model_metadata: ModelMetadata,
    app_tok: AppTokenizer,
    label_max_words: int,
    max_examples: int,
) -> str:
    md = Md()
    md.p("A neural network component has been analyzed from two perspectives.")
    md.extend(_component_header(component, model_metadata))

    md.h(2, "Activation examples — where the component fires")
    md.extend(build_fires_on_examples(component, app_tok, max_examples))

    md.h(2, "Activation examples — what the model produces")
    md.extend(build_says_examples(component, app_tok, max_examples))

    md.h(2, "Two-perspective analysis")
    md.p(
        f'OUTPUT FUNCTION: "{output_label.label}" (confidence: {output_label.confidence})\n'
        f"  Reasoning: {output_label.reasoning}\n\n"
        f'INPUT FUNCTION: "{input_label.label}" (confidence: {input_label.confidence})\n'
        f"  Reasoning: {input_label.reasoning}"
    )

    md.h(2, "Task")
    md.p(
        f"Synthesize these into a single unified label (max {label_max_words} words) "
        "that captures the component's complete role. If input and output suggest the "
        "same concept, unify them. If they describe genuinely different aspects "
        "(e.g. fires on X, produces Y), combine both."
    )
    md.p(JSON_INSTRUCTION)

    return md.build()


def _format_related(
    components: list[RelatedComponent],
    model_metadata: ModelMetadata,
    app_tok: AppTokenizer,
) -> Md:
    visible = [n for n in components if n.label is not None or _is_token_entry(n.component_key)]
    md = Md()
    if not visible:
        md.p("(no related components with labels found)")
        return md

    max_attr = max(abs(n.attribution) for n in visible)
    norm = max_attr if max_attr > 0 else 1.0

    lines: list[str] = []
    for n in visible:
        display = _component_display(n.component_key, model_metadata, app_tok)
        rel_attr = n.attribution / norm

        parts = [f"  {display} (relative attribution: {rel_attr:+.2f}"]
        if n.pmi is not None:
            parts.append(f", co-firing PMI: {n.pmi:.2f}")
        parts.append(")")

        line = "".join(parts)
        if n.label is not None:
            line += f'\n    label: "{n.label}" (confidence: {n.confidence})'
        lines.append(line)

    md.p("\n".join(lines))
    return md


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
