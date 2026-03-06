"""Dual-view interpretation strategy.

Key differences from compact_skeptical:
- Output token data presented first
- Two example sections: "fires on" (current token) and "produces" (next token)
- Human-readable layer descriptions with position context
- Task framing asks for functional description, not detection label
"""

from spd.app.backend.app_tokenizer import AppTokenizer
from spd.autointerp.config import DualViewConfig
from spd.autointerp.prompt_helpers import (
    DATASET_DESCRIPTIONS,
    build_fires_on_examples,
    build_input_section,
    build_output_section,
    build_says_examples,
    density_note,
    human_layer_desc,
    layer_position_note,
)
from spd.autointerp.schemas import ModelMetadata
from spd.harvest.analysis import TokenPRLift
from spd.harvest.schemas import ComponentData


def format_prompt(
    config: DualViewConfig,
    component: ComponentData,
    model_metadata: ModelMetadata,
    app_tok: AppTokenizer,
    input_token_stats: TokenPRLift,
    output_token_stats: TokenPRLift,
) -> str:
    input_pmi: list[tuple[str, float]] | None = None
    output_pmi: list[tuple[str, float]] | None = None

    if config.include_pmi:
        input_pmi = (
            [(app_tok.get_tok_display(tid), pmi) for tid, pmi in component.input_token_pmi.top]
            if component.input_token_pmi.top
            else None
        )
        output_pmi = (
            [(app_tok.get_tok_display(tid), pmi) for tid, pmi in component.output_token_pmi.top]
            if component.output_token_pmi.top
            else None
        )

    output_section = build_output_section(output_token_stats, output_pmi)
    input_section = build_input_section(input_token_stats, input_pmi)
    fires_on_examples = build_fires_on_examples(component, app_tok, config.max_examples)
    says_examples = build_says_examples(component, app_tok, config.max_examples)

    if component.firing_density > 0.0:
        rate_str = f"~1 in {int(1 / component.firing_density)} tokens"
    else:
        rate_str = "extremely rare"

    canonical = model_metadata.layer_descriptions.get(component.layer, component.layer)
    layer_desc = human_layer_desc(canonical, model_metadata.n_blocks)
    position_note = layer_position_note(canonical, model_metadata.n_blocks)
    dens_note = density_note(component.firing_density)

    context_notes = " ".join(filter(None, [position_note, dens_note]))

    dataset_line = ""
    if config.include_dataset_description:
        dataset_desc = DATASET_DESCRIPTIONS.get(
            model_metadata.dataset_name, model_metadata.dataset_name
        )
        dataset_line = f", dataset: {dataset_desc}"

    forbidden_sentence = (
        "FORBIDDEN vague words: " + ", ".join(config.forbidden_words) + ". "
        if config.forbidden_words
        else ""
    )

    return f"""\
Describe what this neural network component does.

Each component is a learned linear transformation inside a weight matrix. It has an input function (what causes it to fire) and an output function (what tokens it causes the model to produce). These are often different — a component might fire on periods but produce sentence-opening words, or fire on prepositions but produce abstract nouns.

Consider all of the evidence below critically. Token statistics can be noisy, especially for high-density components. The activation examples are sampled and may not be representative. Look for patterns that are consistent across multiple sources of evidence.

## Context
- Model: {model_metadata.model_class} ({model_metadata.n_blocks} blocks){dataset_line}
- Component location: {layer_desc}
- Component firing rate: {component.firing_density * 100:.2f}% ({rate_str})

{context_notes}

## Output tokens (what the model produces when this component fires)

{output_section}
## Input tokens (what causes this component to fire)

{input_section}
## Activation examples — where the component fires

<<delimiters>> mark tokens where this component is active.

{fires_on_examples}
## Activation examples — what the model produces

Same examples with <<delimiters>> shifted right by one — showing the token that follows each firing position.

{says_examples}

## Task

Give a {config.label_max_words}-word-or-fewer label describing this component's function. The label should read like a short description of the job this component does in the network. Use both the input and output evidence.

Examples of good labels across different component types:
- "word stem completion (stems → suffixes)"
- "closes dialogue with quotation marks"
- "object pronouns after verbs"
- "story-ending moral resolution vocabulary"
- "aquatic scene vocabulary (frog, river, pond)"
- "'of course' and abstract nouns after prepositions"

Say "unclear" if the evidence is too weak or diffuse. {forbidden_sentence}Lowercase only.
"""
