"""Compact skeptical interpretation strategy.

Short labels (2-5 words), skeptical tone, structured JSON output.
Extracted from the original prompt_template.py.
"""

from spd.app.backend.app_tokenizer import AppTokenizer
from spd.app.backend.utils import delimit_tokens
from spd.autointerp.config import CompactSkepticalConfig
from spd.autointerp.schemas import ModelMetadata
from spd.harvest.analysis import TokenPRLift
from spd.harvest.schemas import ComponentData

DATASET_DESCRIPTIONS: dict[str, str] = {
    "SimpleStories/SimpleStories": (
        "SimpleStories: 2M+ short stories (200-350 words), grade 1-8 reading level. "
        "Simple vocabulary, common narrative elements."
    ),
    "danbraunai/pile-uncopyrighted-tok-shuffled": (
        "The Pile (uncopyrighted subset): diverse English text from books, "
        "academic papers, code, web pages, and other sources."
    ),
}

SPD_CONTEXT = (
    "Each component has a causal importance (CI) value per token position. "
    "High CI (near 1) = essential, cannot be ablated. Low CI (near 0) = ablatable."
)


def format_prompt(
    config: CompactSkepticalConfig,
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

    input_section = _build_input_section(input_token_stats, input_pmi)
    output_section = _build_output_section(output_token_stats, output_pmi)
    examples_section = _build_examples_section(
        component,
        app_tok,
        config.max_examples,
    )

    if component.firing_density > 0.0:
        rate_str = f"~1 in {int(1 / component.firing_density)} tokens"
    else:
        rate_str = "extremely rare"  # TODO(oli) make this string better. does this even happen?

    layer_desc = model_metadata.layer_descriptions.get(component.layer, component.layer)

    dataset_line = ""
    if config.include_dataset_description:
        dataset_desc = DATASET_DESCRIPTIONS[model_metadata.dataset_name]
        dataset_line = f", dataset: {dataset_desc}"

    spd_context_block = f"\n{SPD_CONTEXT}\n" if config.include_spd_context else ""

    forbidden = ", ".join(config.forbidden_words) if config.forbidden_words else "(none)"

    return f"""\
Label this neural network component.
{spd_context_block}
## Context
- Model: {model_metadata.model_class} ({model_metadata.n_blocks} blocks){dataset_line}
- Component location: {layer_desc}
- Component firing rate: {component.firing_density * 100:.2f}% ({rate_str})

## Token correlations

{input_section}
{output_section}

## Activation examples (active tokens in <<delimiters>>)

{examples_section}

## Task

Give a 2-{config.label_max_words} word label for what this component detects.

Be SKEPTICAL. If you can't identify specific tokens or a tight grammatical pattern, say "unclear".

Rules:
1. Good labels name SPECIFIC tokens: "'the'", "##ing suffix", "she/her pronouns"
2. Say "unclear" if: tokens are too varied, pattern is abstract, or evidence is weak
3. FORBIDDEN words (too vague): {forbidden}
4. Lowercase only
5. Confidence: "high" = clear, specific pattern with strong evidence; "medium" = plausible but noisy; "low" = speculative

GOOD: "##ed suffix", "'and' conjunction", "she/her/hers", "period then capital", "unclear"
BAD: "various words and punctuation", "verbs and adjectives", "tokens near commas"
"""


def _build_input_section(
    input_stats: TokenPRLift,
    input_pmi: list[tuple[str, float]] | None,
) -> str:
    section = ""

    if input_stats.top_recall:
        section += "**Input tokens with highest recall (most common current tokens when the component is firing)**\n"
        for tok, recall in input_stats.top_recall[:8]:
            section += f"- {repr(tok)}: {recall * 100:.0f}%\n"

    if input_stats.top_precision:
        section += "\n**Input tokens with highest precision (probability the component fires given the current token is X)**\n"
        for tok, prec in input_stats.top_precision[:8]:
            section += f"- {repr(tok)}: {prec * 100:.0f}%\n"

    if input_pmi:
        section += "\n**Input tokens with highest PMI (pointwise mutual information. Tokens with higher-than-base-rate likelihood of co-occurrence with the component firing)**\n"
        for tok, pmi in input_pmi[:6]:
            section += f"- {repr(tok)}: {pmi:.2f}\n"

    return section


def _build_output_section(
    output_stats: TokenPRLift,
    output_pmi: list[tuple[str, float]] | None,
) -> str:
    section = ""

    if output_stats.top_precision:
        section += "**Output precision — of all predicted probability for token X, what fraction is at positions where this component fires?**\n"
        for tok, prec in output_stats.top_precision[:10]:
            section += f"- {repr(tok)}: {prec * 100:.0f}%\n"

    if output_pmi:
        section += "\n**Output PMI — tokens the model predicts at higher-than-base-rate when this component fires:**\n"
        for tok, pmi in output_pmi[:6]:
            section += f"- {repr(tok)}: {pmi:.2f}\n"

    return section


def _build_examples_section(
    component: ComponentData,
    app_tok: AppTokenizer,
    max_examples: int,
) -> str:
    section = ""
    examples = component.activation_examples[:max_examples]

    for i, ex in enumerate(examples):
        if any(ex.firings):
            spans = app_tok.get_spans(ex.token_ids)
            tokens = list(zip(spans, ex.firings, strict=True))
            section += f"{i + 1}. {delimit_tokens(tokens)}\n"

    return section
