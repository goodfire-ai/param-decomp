"""Prompt templates for component auto-interpretation."""

import json

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from spd.app.backend.utils import build_token_lookup
from spd.autointerp import MAX_EXAMPLES_PER_COMPONENT
from spd.autointerp.schemas import InterpContext
from spd.harvest.analysis import TokenPRLift
from spd.harvest.schemas import ComponentData


def _parse_layer_description(layer: str, n_blocks: int) -> str:
    """Parse layer name into human-readable description.

    e.g. "h.2.mlp.c_fc" -> "MLP up-projection in layer 3 of 4"
    """
    parts = layer.split(".")
    assert parts[0] == "h", f"unexpected layer format: {layer}"
    layer_idx = int(parts[1])
    layer_num = layer_idx + 1

    sublayer = ".".join(parts[2:])
    sublayer_desc = {
        "mlp.c_fc": "MLP up-projection",
        "mlp.c_proj": "MLP down-projection",
        "attn.q_proj": "attention Q projection",
        "attn.k_proj": "attention K projection",
        "attn.v_proj": "attention V projection",
        "attn.o_proj": "attention output projection",
    }.get(sublayer, sublayer)

    return f"{sublayer_desc} in layer {layer_num} of {n_blocks}"


INTERPRETATION_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "description": "3-10 word label describing what the component detects/represents",
        },
        "confidence": {
            "type": "string",
            "enum": ["low", "medium", "high"],
            "description": "low = multiple plausible interpretations or weak signal; medium = coherent pattern but some noise; high = clear, consistent pattern across metrics",
        },
        "reasoning": {
            "type": "string",
            "description": "2-4 sentences explaining the evidence and ambiguities",
        },
    },
    "required": ["label", "confidence", "reasoning"],
    "additionalProperties": False,
}

INTERPRETATION_SCHEMA_JSON_STR = json.dumps(INTERPRETATION_SCHEMA, indent=2)


DATASET_DESCRIPTIONS: dict[str, str] = {
    "SimpleStories/SimpleStories": """\
SimpleStories is a dataset of 2M+ short stories (200-350 words each) at a grade 1-8 reading level.
The stories cover diverse themes (friendship, courage, loss, discovery) and settings (magical lands,
schools, forests, space). The vocabulary is simple, everyday English. Stories feature common narrative
elements: characters with names, emotions, dialogue, and simple plot arcs with resolutions.""",
}

SPD_THEORETICAL_CONTEXT = """\
SPD (Stochastic Parameter Decomposition) decomposes a neural network's weight matrices into rank-1
"subcomponents". Each subcomponent has a causal importance (CI) value predicted *per sequence position*
by a small auxiliary neural network. CI indicates how necessary the component is for the model's output
at that position: high CI (close to 1) means the component is essential and cannot be ablated; low CI
(close to 0) means it can be removed without affecting output. The training objective encourages
sparsity: as few components as possible should have high CI for any given input."""


def _build_data_presentation_section(arch: InterpContext) -> str:
    """Explain how the model sees data and how activation examples are constructed."""
    k = arch.context_tokens_per_side
    window_size = 2 * k + 1
    return f"""\
## Data Presentation

The model processes sequences of {arch.seq_len} tokens. Each activation example below shows a \
{window_size}-token window centered on the token where the component fired, with up to {k} tokens \
of context on each side. Windows are truncated at sequence boundaries."""


def format_prompt_template(
    component: ComponentData,
    arch: InterpContext,
    tokenizer: PreTrainedTokenizerBase,
    input_token_stats: TokenPRLift,
    output_token_stats: TokenPRLift,
    ci_display_threshold: float = 0.3,
    output_precision_top_k: int = 40,
) -> str:
    """Improved prompt template using recall/precision/PMI.

    Key improvements over v1:
    - Uses recall AND precision for input tokens
    - Uses precision AND PMI for output tokens
    - Only shows high-CI tokens in examples (reduces noise)
    - Includes inline metric definitions
    - Better dataset descriptions to avoid vacuous interpretations
    """
    lookup = build_token_lookup(tokenizer, tokenizer.name_or_path)
    PADDING_SENTINEL = -1

    # Convert PMI from ComponentData to decoded tokens
    input_pmi = (
        [(lookup[tid], pmi) for tid, pmi in component.input_token_pmi.top]
        if component.input_token_pmi.top
        else None
    )
    output_pmi = (
        [(lookup[tid], pmi) for tid, pmi in component.output_token_pmi.top]
        if component.output_token_pmi.top
        else None
    )

    # Build input token section using recall, precision, and PMI
    input_section = _build_input_token_section(input_token_stats, input_pmi)

    # Build output token section using precision and PMI
    output_section = _build_output_token_section(
        output_token_stats, output_pmi, output_precision_top_k
    )

    # Build examples showing only high-CI tokens
    examples_section = _build_examples_section(
        component, tokenizer, lookup, ci_display_threshold, PADDING_SENTINEL
    )

    # Get dataset description
    dataset_description = DATASET_DESCRIPTIONS.get(
        arch.dataset_name, f"Dataset: {arch.dataset_name}"
    )

    # Calculate firing rate context
    firing_rate_context = ""
    if component.mean_ci > 0:
        tokens_per_firing = int(1 / component.mean_ci)
        firing_rate_context = f" (fires on ~1 in {tokens_per_firing} tokens)"

    layer_desc = _parse_layer_description(component.layer, arch.n_blocks)

    data_presentation = _build_data_presentation_section(arch)

    return f"""\
Label this neural network component from a Stochastic Parameter Decomposition.

## Background

{SPD_THEORETICAL_CONTEXT}

## Model Context

**Model**: {arch.model_class} ({arch.n_blocks} layers)
**Dataset**: {dataset_description}

{data_presentation}

## Component Context

**Component location**: {layer_desc}
**Activation rate**: {component.mean_ci * 100:.2f}%{firing_rate_context}

---

{input_section}

---

{output_section}

---

{examples_section}

---

## Task

Based on the above context, what concept or pattern does this component represent?
Consider both what the component *does* (what tokens it helps predict) and what triggers it.

If the pattern is unclear or the evidence is weak, say so. Use "unclear" or "noisy" in your label if appropriate—do not force an interpretation where none exists.

Return JSON:
```json
{INTERPRETATION_SCHEMA_JSON_STR}
```
"""


def _build_input_token_section(
    input_stats: TokenPRLift,
    input_pmi: list[tuple[str, float]] | None,
) -> str:
    """Build input token analysis section using recall, precision, and PMI."""
    section = """\
## Correlations with Input Tokens

The following metrics concern correlations between this component firing and the "current" token (the token at the position where the component is active).

"""

    # Recall section
    if input_stats.top_recall:
        section += '**Recall:** _"What % of this component\'s firings occurred on token X?"_\n'
        for token, recall in input_stats.top_recall[:10]:
            pct = recall * 100
            if pct >= 1:
                section += f"  {repr(token)}: {pct:.0f}%\n"
            elif pct >= 0.1:
                section += f"  {repr(token)}: {pct:.1f}%\n"
        section += "\n"

    # Precision section - very important for detecting deterministic triggers
    if input_stats.top_precision:
        section += '**Precision:** _"When token X appears, what % of the time does this component fire?"_\n'
        for token, prec in input_stats.top_precision[:10]:
            section += f"  {repr(token)}: {prec * 100:.0f}%\n"
        section += "\n"

    # PMI section - shows surprising associations
    if input_pmi:
        section += "**PMI:** _Tokens with higher-than-expected co-occurrence_\n"
        for token, pmi in input_pmi[:8]:
            section += f"  {repr(token)}: {pmi:.2f}\n"
        section += "\n"

    return section


def _build_output_token_section(
    output_stats: TokenPRLift,
    output_pmi: list[tuple[str, float]] | None,
    top_k: int,
) -> str:
    """Build output token analysis section using precision and PMI."""
    section = """\
## Correlations with Predicted Tokens

The following metrics concern correlations between this component firing and the token the model predicts at that position.

"""

    # Precision section
    if output_stats.top_precision:
        section += '**Precision:** _"When the model predicts token X, what % of the time is this component active?"_\n\n'
        # Group by precision ranges
        very_high = [(t, p) for t, p in output_stats.top_precision[:top_k] if p > 0.90]
        high = [(t, p) for t, p in output_stats.top_precision[:top_k] if 0.70 <= p <= 0.90]
        medium = [(t, p) for t, p in output_stats.top_precision[:top_k] if 0.50 <= p < 0.70]

        if very_high:
            tokens = [repr(t) for t, _ in very_high[:30]]
            section += f"**Very high (>90%)**: {', '.join(tokens)}\n\n"

        if high:
            tokens = [repr(t) for t, _ in high[:20]]
            section += f"**High (70-90%)**: {', '.join(tokens)}\n\n"

        if medium:
            tokens = [repr(t) for t, _ in medium[:15]]
            section += f"**Medium (50-70%)**: {', '.join(tokens)}\n\n"

        if not very_high and not high and not medium:
            tokens_with_prec = [
                f"{repr(t)} ({p * 100:.0f}%)" for t, p in output_stats.top_precision[:15]
            ]
            section += f"Top by precision: {', '.join(tokens_with_prec)}\n\n"

    # PMI section for output tokens
    if output_pmi:
        section += "**PMI:** _Tokens with higher-than-expected co-occurrence_\n"
        for token, pmi in output_pmi[:10]:
            section += f"  {repr(token)}: {pmi:.2f}\n"

    return section


def _build_examples_section(
    component: ComponentData,
    tokenizer: PreTrainedTokenizerBase,
    lookup: dict[int, str],
    ci_threshold: float,
    padding_sentinel: int,
) -> str:
    """Build examples section showing only high-CI tokens."""
    section = f"""\
## Activation Examples

_Showing tokens where CI > {ci_threshold} (component is active)_

"""
    examples = component.activation_examples[:MAX_EXAMPLES_PER_COMPONENT]
    for i, example in enumerate(examples):
        # Decode full text
        valid_tokens = [t for t in example.token_ids if t != padding_sentinel and t >= 0]
        full_text = tokenizer.decode(valid_tokens) if valid_tokens else ""
        display_text = full_text.replace("\n", " ")

        # Get high-CI tokens with their CI values
        active_tokens = []
        for tid, ci in zip(example.token_ids, example.ci_values, strict=True):
            if ci > ci_threshold and tid != padding_sentinel and tid >= 0:
                tok = lookup[tid].strip()
                active_tokens.append(f'"{tok}" ({ci:.2f})')

        active_str = ", ".join(active_tokens)

        section += f'Ex {i + 1}: "{display_text}"\n'
        section += f"  Active tokens: {active_str}\n\n"

    return section
