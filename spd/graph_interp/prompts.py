"""Prompt formatters for graph interpretation.

Three prompts, all using canon-style evidence presentation (XML examples with CI/act
annotations, recall + PMI token stats):

1. Output pass (late→early): focused on what the component contributes to predictions
2. Input pass (early→late): focused on what triggers the component
3. Unification: synthesizes output + input labels into a unified description
"""

from spd.app.backend.app_tokenizer import AppTokenizer
from spd.autointerp.config import CANON_RENDERING
from spd.autointerp.prompt_helpers import (
    build_annotated_examples,
    token_stats_section,
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
        "reasoning": {"type": "string"},
        "summary_for_neighbors": {"type": "string"},
    },
    "required": ["label", "reasoning", "summary_for_neighbors"],
    "additionalProperties": False,
}

UNIFIED_LABEL_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "label": {"type": "string"},
        "reasoning": {"type": "string"},
    },
    "required": ["label", "reasoning"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------


def _spd_preamble() -> Md:
    md = Md()
    md.p(
        "In Stochastic Parameter Decomposition, each weight matrix is decomposed into "
        "rank-1 subcomponents parameterised as U \u2022 V (dimensions `d_out` \u00d7 `d_in`). "
        "Each represents a one-dimensional slice of the computation the weight matrix performs."
    )
    md.p(
        "At each token position, each component has a **Causal Importance (CI)** value "
        "in (0, 1) measuring how much ablating it would change the model's output. "
        'A component "fires" when CI exceeds a threshold. CI is the primary signal \u2014 '
        "weight high-CI positions heavily."
    )
    md.p(
        "Components also have an **inner activation (act)** \u2014 the alignment of the "
        "input with the read direction. **Relative sign is meaningful**: positive-act and "
        "negative-act tokens produce opposite contributions, so a sign split across examples "
        "indicates two distinct roles."
    )
    return md


def _component_header(
    component: ComponentData,
    model_metadata: ModelMetadata,
) -> Md:
    canonical = model_metadata.layer_descriptions.get(component.layer, component.layer)
    layer_desc = _short_layer_desc(canonical)
    rate_str = (
        f"~1 in {int(1 / component.firing_density)} tokens"
        if component.firing_density > 0.0
        else "extremely rare"
    )
    md = Md()
    md.h(3, "This component")
    md.p(
        f"Model: {model_metadata.n_blocks}-block transformer. "
        f"This component is in the {layer_desc}. "
        f"Firing rate: {component.firing_density * 100:.2f}% ({rate_str})."
    )
    return md


def _short_layer_desc(canonical: str) -> str:
    """Compact layer description: 'attn key proj, layer 2' instead of
    'attention key projection in the 2nd of 4 blocks'."""
    parts = canonical.split(".")
    if len(parts) != 3:
        return canonical
    block, module, proj = parts
    block_num = int(block) + 1
    module_names = {"attn": "attention", "mlp": "MLP"}
    proj_names = {
        "k": "key proj",
        "q": "query proj",
        "v": "value proj",
        "o": "output proj",
        "up": "up-proj",
        "down": "down-proj",
    }
    mod = module_names.get(module, module)
    p = proj_names.get(proj, proj)
    return f"{mod} {p}, layer {block_num}"


def _examples_section(
    component: ComponentData,
    app_tok: AppTokenizer,
    max_examples: int,
    context_tokens_per_side: int,
) -> Md:
    md = Md()
    md.h(3, "Activating examples")
    md.p(
        "Sampled positions where the component fires (CI above threshold), shown as "
        f"a window of up to {context_tokens_per_side} tokens of context on each side. "
        "Each example has a `<raw>` view and an `<annotated>` view with firing tokens "
        "wrapped as `[[[token (ci:X, act:Y)]]]`."
    )
    md.extend(build_annotated_examples(component, app_tok, max_examples, rendering=CANON_RENDERING))
    return md


def _related_section(
    related: list[RelatedComponent],
    heading: str,
    description: str,
    model_metadata: ModelMetadata,
    app_tok: AppTokenizer,
) -> Md:
    md = Md()
    md.h(3, heading)
    md.p(description)
    md.extend(_format_related(related, model_metadata, app_tok))
    return md


# ---------------------------------------------------------------------------
# Output pass prompt
# ---------------------------------------------------------------------------


def format_output_prompt(
    component: ComponentData,
    model_metadata: ModelMetadata,
    app_tok: AppTokenizer,
    output_token_stats: TokenPRLift,
    related: list[RelatedComponent],
    label_max_words: int,
    max_examples: int,
    context_tokens_per_side: int,
) -> str:
    md = Md()
    md.p(
        "You are labeling a neural network component's **output function** \u2014 "
        "what it contributes to the model's predictions when it fires."
    )
    md.extend(_spd_preamble())
    md.extend(_component_header(component, model_metadata))

    md.h(2, "Evidence")
    md.extend(
        token_stats_section(
            output_token_stats, "Output", "what the model produces when this component fires"
        )
    )
    md.extend(_examples_section(component, app_tok, max_examples, context_tokens_per_side))
    md.extend(
        _related_section(
            related,
            "Downstream components",
            "Components in later layers with the strongest gradient attribution from this one. "
            "Positive attribution means this component strengthens/enables the downstream one; "
            "negative means it inhibits/counteracts it.",
            model_metadata,
            app_tok,
        )
    )

    md.h(2, "Task")
    md.p(
        f"Give a {label_max_words}-word-or-fewer label describing this component's "
        "output function. The label should describe the most salient aspect of what this "
        "component contributes to predictions \u2014 this could be specific output tokens it "
        "promotes, what tokens tend to follow its activations, or what downstream components "
        "it enables or inhibits. Lead with whatever is most distinctive."
    )
    md.p(
        "Please also provide your reasoning, and a concise `summary_for_neighbors` "
        "(1\u20132 sentences) explaining what this component does \u2014 this will be shown "
        "to neighboring components during their labeling to help them understand the graph context. "
        "Be epistemically honest \u2014 express uncertainty when the evidence is weak. "
        "Lowercase only."
    )
    md.p('Respond with JSON: {"label": "...", "reasoning": "...", "summary_for_neighbors": "..."}')

    return md.build()


# ---------------------------------------------------------------------------
# Input pass prompt
# ---------------------------------------------------------------------------


def format_input_prompt(
    component: ComponentData,
    model_metadata: ModelMetadata,
    app_tok: AppTokenizer,
    input_token_stats: TokenPRLift,
    related: list[RelatedComponent],
    label_max_words: int,
    max_examples: int,
    context_tokens_per_side: int,
) -> str:
    md = Md()
    md.p(
        "You are labeling a neural network component's **input function** \u2014 "
        "what tokens and contexts it responds to."
    )
    md.extend(_spd_preamble())
    md.extend(_component_header(component, model_metadata))

    md.h(2, "Evidence")
    md.extend(
        token_stats_section(
            input_token_stats, "Input", "tokens at positions where this component fires"
        )
    )
    md.extend(_examples_section(component, app_tok, max_examples, context_tokens_per_side))
    md.extend(
        _related_section(
            related,
            "Upstream components",
            "Components in earlier layers with the strongest gradient attribution to this one. "
            "Positive attribution means the upstream component strengthens/enables this one; "
            "negative means it inhibits/counteracts it.",
            model_metadata,
            app_tok,
        )
    )

    md.h(2, "Task")
    md.p(
        f"Give a {label_max_words}-word-or-fewer label describing this component's "
        "input function. The label should describe the most salient aspect of what "
        "this component responds to \u2014 this could be specific input tokens, broader "
        "contexts or text domains, positional patterns, or what upstream components "
        "feed into it. Lead with whatever is most distinctive."
    )
    md.p(
        "Please also provide your reasoning, and a concise `summary_for_neighbors` "
        "(1\u20132 sentences) explaining what this component responds to \u2014 this will be shown "
        "to neighboring components during their labeling to help them understand the graph context. "
        "Be epistemically honest \u2014 express uncertainty when the evidence is weak. "
        "Lowercase only."
    )
    md.p('Respond with JSON: {"label": "...", "reasoning": "...", "summary_for_neighbors": "..."}')

    return md.build()


# ---------------------------------------------------------------------------
# Unification prompt
# ---------------------------------------------------------------------------


def format_unification_prompt(
    output_label: LabelResult,
    input_label: LabelResult,
    component: ComponentData,
    model_metadata: ModelMetadata,
    app_tok: AppTokenizer,
    output_token_stats: TokenPRLift | None,
    input_token_stats: TokenPRLift | None,
    label_max_words: int,
    max_examples: int,
    context_tokens_per_side: int,
) -> str:
    md = Md()
    md.p("A neural network component has been analyzed from two perspectives.")
    md.extend(_spd_preamble())
    md.extend(_component_header(component, model_metadata))

    md.h(2, "Evidence")
    if output_token_stats is not None:
        md.extend(
            token_stats_section(
                output_token_stats, "Output", "what the model produces when this component fires"
            )
        )
    if input_token_stats is not None:
        md.extend(
            token_stats_section(
                input_token_stats, "Input", "tokens at positions where this component fires"
            )
        )
    md.extend(_examples_section(component, app_tok, max_examples, context_tokens_per_side))

    md.h(2, "Two-perspective analysis")
    md.p(f'**Output function:** "{output_label.label}"\n  Reasoning: {output_label.reasoning}')
    md.p(f'**Input function:** "{input_label.label}"\n  Reasoning: {input_label.reasoning}')

    md.h(2, "Task")
    md.p(
        f"Synthesize these into a single unified label (max {label_max_words} words) "
        "that captures the component's most salient behavior. If input and output describe "
        "the same concept, unify them. If they describe genuinely different aspects, "
        "combine both. Lead with whatever is most distinctive."
    )
    md.p(
        "Be epistemically honest \u2014 express uncertainty when the evidence is mixed. "
        "Lowercase only."
    )
    md.p('Respond with JSON: {"label": "...", "reasoning": "..."}')

    return md.build()


# ---------------------------------------------------------------------------
# Related-component formatting
# ---------------------------------------------------------------------------


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

    pos = [n for n in visible if n.attribution >= 0]
    neg = [n for n in visible if n.attribution < 0]

    max_attr = max(abs(n.attribution) for n in visible)
    norm = max_attr if max_attr > 0 else 1.0

    def fmt(n: RelatedComponent) -> str:
        display = _component_display(n.component_key, model_metadata, app_tok)
        rel = n.attribution / norm
        pmi_str = f", cofiring PMI {n.pmi:.1f}" if n.pmi is not None else ""
        if n.label is not None:
            line = f'- "{n.label}" \u2014 {display} ({rel:+.2f}{pmi_str})'
            if n.summary:
                line += f"\n  {n.summary}"
            return line
        return f"- {display} ({rel:+.2f}{pmi_str})"

    if pos:
        md.p("**Positive** (strengthening):")
        md.p("\n".join(fmt(n) for n in pos))
    if neg:
        md.p("**Negative** (inhibiting):")
        md.p("\n".join(fmt(n) for n in neg))

    return md


def _is_token_entry(key: str) -> bool:
    layer = key.rsplit(":", 1)[0]
    return layer in ("embed", "output")


def _component_display(key: str, model_metadata: ModelMetadata, app_tok: AppTokenizer) -> str:
    layer, idx_str = key.rsplit(":", 1)
    match layer:
        case "embed":
            return f'embed("{app_tok.get_tok_display(int(idx_str))}")'
        case "output":
            return f'output("{app_tok.get_tok_display(int(idx_str))}")'
        case _:
            canonical = model_metadata.layer_descriptions.get(layer, layer)
            return f"{_short_layer_desc(canonical)} #{idx_str}"
